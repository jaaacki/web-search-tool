# Agent policy

## Facts an agent cannot guess
- **PRCI.** Pull requests targeting `main` run the pinned API dependency check and fail-fast assert scripts in Python 3.15 (`.woodpecker/pr.yml`). Packaged Docker-image regression remains a separate required check; CI does not deploy. Record actual CI and packaged-image evidence in the batch PR body.
- **Deploy is manual** on lab-001 (README "Deploy"). Merging does not ship. Deploying needs the owner's go-ahead.
- **Traefik routes live in `sparkfn/lab-001-cluster`**, not here.
- **`searxng/settings.yml` and `SEARXNG_ENGINES` in `api/app.py` must name the same engines.** Check the pinned image's `/config` before adding an engine.
- Runtime data lives under `./data` (gitignored). Never commit `.env`.

## Git history is the record
- Start from a GitHub issue; create one if none exists. Branch `fix|feat|chore/<issue>-<slug>` from `main`.
- PR to `main`, with `Fixes #N` and the validation evidence in the body. Squash-merge, then delete the branch.
- Plans, status and decisions go in the issue or PR. No `PLAN.md`, `STATUS.md`, `NOTES.md` or similar breadcrumbs. Repo markdown is only for stable docs (`README.md`, this file).
- When several changes are in flight, merge them into one batch branch and open one PR for the batch.
