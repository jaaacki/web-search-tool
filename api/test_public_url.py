"""Public URL/IP boundary checks for #51, run without DNS or provider traffic."""

import asyncio
import ipaddress

import app


def run(coro):
    return asyncio.run(coro)


def test_malformed_urls_fail_cold_and_warm_dns_cache():
    saved_resolver = app._host_resolves_public
    saved_cache = app._dns_cache.copy()

    def resolve_public(_host, _port):
        return True

    async def check():
        app._dns_cache.clear()
        app._host_resolves_public = resolve_public
        assert await app.is_allowed_public_url("https://example.com:bad/path") is False
        assert not app._dns_cache, "invalid port must fail before hostname-cache lookup"
        assert await app.is_allowed_public_url("https://example.com/") is True
        assert await app.is_allowed_public_url("https://example.com:bad/path") is False
        assert await app.is_allowed_public_url("https://example.com:99999/path") is False
        assert await app.is_allowed_public_url("http://[2001:db8::1") is False
        assert await app.is_allowed_public_url("http://example.com／evil") is False
        assert await app.is_allowed_public_url("https://example.org/") is True

    try:
        run(check())
    finally:
        app._host_resolves_public = saved_resolver
        app._dns_cache.clear()
        app._dns_cache.update(saved_cache)


def test_only_global_unicast_addresses_are_public():
    public = ("8.8.8.8", "2001:4860:4860::8888", "::ffff:8.8.8.8")
    blocked = (
        "100.64.0.1",
        "::ffff:100.64.0.1",
        "192.168.1.1",
        "127.0.0.1",
        "169.254.1.1",
        "224.0.0.1",
        "192.0.2.1",
        "fec0::1",
        "::",
    )
    for address in public:
        assert app.is_public_ip(ipaddress.ip_address(address)), address
    for address in blocked:
        assert not app.is_public_ip(ipaddress.ip_address(address)), address


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
