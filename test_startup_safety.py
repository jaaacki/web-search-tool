"""Offline Docker/Compose regression for #54. Run: python3 test_startup_safety.py IMAGE.

IMAGE must be built from api/Dockerfile's runtime target. Only private temporary
config/env files, an isolated internal network, dummy keys and mock upstreams
are used. No host ports, real providers, production containers or reboot.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
COMPOSE_VARIABLES = set(re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", (ROOT / "docker-compose.yml").read_text()))

# Loaded by the unchanged two-worker runtime CMD; every upstream is in-memory.
MOCK_UPSTREAMS = '''
import json
import httpx

rows = [
    {"url": "https://93.184.216.34/a", "title": "Unrelated", "description": "An unrelated page."},
    {"url": "https://93.184.216.34/b", "title": "Startup safety", "description": "cloud brave startup safety " * 20},
]

def handle(request):
    host, path = request.url.host, request.url.path
    if host in {"api.firecrawl.dev", "api.search.brave.com"}:
        with open("/tmp/provider-calls", "a") as calls:
            calls.write(host + path + "\\n")
    if host == "api.firecrawl.dev" and path == "/v2/search":
        if json.loads(request.content)["query"] == "cloud":
            return httpx.Response(200, json={"success": True, "data": {"web": rows}})
    if host == "api.firecrawl.dev" and path == "/v2/scrape":
        url = json.loads(request.content)["url"]
        return httpx.Response(200, json={"success": True, "data": {
            "markdown": rows[int(url.endswith("/b"))]["description"],
            "metadata": {"sourceURL": url, "url": url, "statusCode": 200},
        }})
    if host == "api.search.brave.com" and request.url.params.get("q") == "brave":
        return httpx.Response(200, json={"web": {"results": rows}})
    return httpx.Response(503)

original_init = httpx.AsyncClient.__init__
def mock_init(self, *args, **kwargs):
    kwargs.setdefault("transport", httpx.MockTransport(handle))
    original_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = mock_init
'''

CLOCK_PROOF = '''
import asyncio, os, stat, sys, time
from unittest.mock import patch
import httpx
import firecrawl_controls as fc

phase = sys.argv[1]
assert stat.S_IMODE(os.stat("/tmp").st_mode) == 0o1777
fs = os.statvfs("/tmp")
assert fs.f_blocks * fs.f_frsize == 16 * 1024 * 1024
calls = []
def handle(request):
    calls.append(request)
    return httpx.Response(429 if phase == "seed" else 200, headers={"Retry-After": "60"})

async def check():
    controls = fc.FirecrawlControls((fc._Budget("test", ("dummy",)),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        if phase == "fresh":
            assert not fc.STATE_PATH.exists() and not fc.LOCK_PATH.exists()
        try:
            result = await controls.request(client, "/v2/search", {"query": "q"}, timeout=2)
        except fc.FirecrawlUnavailable as exc:
            assert phase in {"seed", "share"} and exc.category == "rate_limited", exc.category
            assert 59 <= exc.retry_after <= 60, exc.retry_after
        else:
            assert phase in {"stale", "fresh"} and result.status_code == 200
    assert len(calls) == (0 if phase == "share" else 1), calls
    assert stat.S_IMODE(fc.STATE_PATH.stat().st_mode) == 0o600
    assert stat.S_IMODE(fc.LOCK_PATH.stat().st_mode) == 0o600

# Wall-clock cooldowns are shared across workers and expire when time moves on.
real_time = time.time
offset = 30 * 86400 if phase == "stale" else 0
with patch.object(fc.time, "time", side_effect=lambda: real_time() + offset):
    asyncio.run(check())
print("ok clock", phase)
'''

SOCKET_PROOF = '''
import http.client, json
from pathlib import Path

# One keep-alive connection keeps per-worker health observations deterministic.
connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)
def call(method, path, payload=None, authenticated=True):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["X-API-Key"] = "dummy"
    connection.request(method, path, json.dumps(payload) if payload else None, headers)
    response = connection.getresponse()
    body = json.loads(response.read())
    return response.status, body

def health(status, provider=None):
    before = Path("/tmp/provider-calls").read_text() if Path("/tmp/provider-calls").exists() else ""
    actual, body = call("GET", "/health", authenticated=False)
    assert actual == status, (actual, body)
    after = Path("/tmp/provider-calls").read_text() if Path("/tmp/provider-calls").exists() else ""
    assert after == before, "health launched a provider call"
    if provider:
        checks = body["data"]["checks"]
        assert checks[provider] == "ok", checks
        assert checks["crawl4ai"].startswith("unavailable"), checks
        assert checks["reranker"].startswith("unavailable"), checks
    else:
        assert body["error"]["code"] == "upstream_unavailable", body

health(503)  # Unknown cloud providers must not fabricate cold-start health.
for path, payload in (("/search", {"query": "cloud"}), ("/crawl", {"url": "https://93.184.216.34/a"})):
    status, body = call("POST", path, payload, authenticated=False)
    assert status == 401 and body["error"]["code"] == "unauthorized", body
for query, depth, provider in (("cloud", "advanced", "firecrawl"), ("brave", "basic", "brave")):
    status, body = call("POST", "/search", {"query": query, "depth": depth, "candidates": 2, "max_results": 2})
    assert status == 200, (status, body)
    results = body["data"]["results"]
    assert results and results[0]["url"].endswith("/b"), results
    assert all(result["score"] is None for result in results), results
    health(200, provider)
status, body = call("POST", "/search", {"query": "outage", "depth": "basic", "candidates": 2})
assert status == 503 and body["error"]["code"] == "upstream_unavailable", (status, body)
health(503)
connection.close()
print("ok socket auth, cloud extraction, Brave/basic, lexical fallback, total-outage health")
'''


def run(*command, **kwargs):
    # Exported app settings outrank --env-file; retain unrelated Docker CLI connection/context settings.
    environment = {name: value for name, value in os.environ.items() if name not in COMPOSE_VARIABLES}
    result = subprocess.run(command, text=True, capture_output=True, timeout=60, env=environment, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{command}:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main(image):
    with tempfile.TemporaryDirectory(prefix="websearch-54-") as directory:
        scratch = Path(directory)
        private_env = scratch / "private.env"
        dummy = dict.fromkeys(("SEARXNG_SECRET", "CRAWL4AI_API_TOKEN", "WEBSEARCH_API_KEY", "BRAVE_API_KEYS"), "dummy")
        dummy["FIRECRAWL_BUDGETS_JSON"] = '[{"id":"test","keys":["dummy"]}]'
        private_env.write_text("".join(f"{name}={value}\n" for name, value in dummy.items()))
        private_env.chmod(0o600)
        # Explicit --env-file prevents default .env loading; skip default service env files too.
        base = ["docker", "compose", "--env-file", str(private_env), "-p", scratch.name]
        config = json.loads(run(*base, "-f", str(ROOT / "docker-compose.yml"), "config", "--no-env-resolution", "--format", "json"))
        api = config["services"]["api"]
        # Fail before starting containers if adversarial host exports overrode any dummy credential.
        assert all(api["environment"][name] == dummy[name] for name in dummy if name != "SEARXNG_SECRET")
        assert config["services"]["searxng"]["environment"]["SEARXNG_SECRET"] == dummy["SEARXNG_SECRET"]
        assert config["services"]["crawl4ai"]["environment"]["CRAWL4AI_API_TOKEN"] == dummy["CRAWL4AI_API_TOKEN"]
        print("ok private Compose credentials ignore exported application settings")
        assert all(dep["condition"] == "service_started" for dep in api["depends_on"].values())
        assert api["tmpfs"] == ["/tmp:rw,noexec,nosuid,size=16m,mode=1777"], api["tmpfs"]
        assert set(api["networks"]) == {"internal", "traefik"}
        assert config["networks"]["traefik"]["external"] is True
        for name, service in config["services"].items():
            assert not service.get("ports"), (name, service.get("ports"))
            service.pop("build", None)
            service.pop("container_name", None)
            service.update(image=image, restart="no", volumes=[], env_file=[], networks={"internal": None})
            if name != "api":
                service.update(command=["python", "-c", "import time; time.sleep(3600)"], environment={}, healthcheck={
                    "test": ["CMD", "python", "-c", "raise SystemExit(1)"],
                    "interval": "1s", "timeout": "1s", "retries": 1, "start_period": "0s",
                })
        api["healthcheck"].update(interval="1s", start_period="0s")  # Same readiness command, faster test polling.
        api["env_file"] = [{"path": str(private_env), "required": True}]
        api["environment"].update(PYTHONPATH="/proof", HEALTH_CACHE_TTL="3600")
        mock_code = scratch / "sitecustomize.py"
        mock_code.write_text(MOCK_UPSTREAMS)
        mock_code.chmod(0o644)
        api["volumes"] = [{"type": "bind", "source": str(mock_code), "target": "/proof/sitecustomize.py", "read_only": True}]
        config["networks"] = {"internal": {"internal": True}}
        compose_file = scratch / "compose.json"
        compose_file.write_text(json.dumps(config))
        compose = [*base, "-f", str(compose_file)]
        run(*compose, "config", "--quiet")  # Real engine-supported Compose validation, private inputs only.
        try:
            locals_ = ("searxng", "crawl4ai", "reranker")
            run(*compose, "up", "-d", "--no-build", "--pull", "never", *locals_)
            ids = run(*compose, "ps", "-q", *locals_).split()
            for _ in range(30):
                states = json.loads(run("docker", "inspect", *ids))
                if len(states) == 3 and all(s["State"]["Health"]["Status"] == "unhealthy" for s in states):
                    break
                time.sleep(1)
            else:
                raise AssertionError("local stubs did not become unhealthy")
            run(*compose, "up", "-d", "--no-build", "--pull", "never", "api")
            container = run(*compose, "ps", "-q", "api").strip()
            inspect = json.loads(run("docker", "inspect", container))[0]
            assert inspect["HostConfig"]["Tmpfs"]["/tmp"] == "rw,noexec,nosuid,size=16m,mode=1777"
            for _ in range(30):
                if json.loads(run("docker", "inspect", container))[0]["State"]["Health"]["Status"] == "healthy":
                    break
                time.sleep(1)
            else:
                raise AssertionError("cold API readiness still depends on optional local/provider health")
            assert all(s["State"]["Health"]["Status"] == "unhealthy" for s in json.loads(run("docker", "inspect", *ids)))
            run(*compose, "exec", "-T", "api", "python", "-c",
                'from pathlib import Path; assert not Path("/tmp/provider-calls").exists()')
            print("ok cold Docker readiness before any provider observation or search; optional locals unhealthy")
            for phase in ("seed", "share", "stale"):
                print(run(*compose, "exec", "-T", "api", "python", "-c", CLOCK_PROOF, phase), end="")
            run(*compose, "restart", "api")
            assert run(*compose, "ps", "-q", "api").strip() == container
            print(run(*compose, "exec", "-T", "api", "python", "-c", CLOCK_PROOF, "fresh"), end="")
            for _ in range(30):
                if run(*compose, "logs", "api").count("Application startup complete.") >= 4:
                    break
                time.sleep(1)
            else:
                raise AssertionError("both default Uvicorn workers did not start before and after restart")
            print(run(*compose, "exec", "-T", "api", "python", "-c", SOCKET_PROOF), end="")
            print("ok Compose: unhealthy optional services do not gate the default two-worker API")
        finally:
            run(*compose, "down", "--volumes", "--remove-orphans")


if __name__ == "__main__":
    main(sys.argv[1])
