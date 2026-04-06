<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Bot Policy — xPyD-proxy

## Language
- **English only** — all code, docs, issues, PRs, comments on GitHub must be in English. No Chinese characters.

## Branch Rules
- **Never push directly to main.** All changes go through PR.
- **Never force push.** If branch is too messy, close PR and open new one.
- Branch from latest main. Keep branch up-to-date by merging main into it.
- Each PR must be independent — no stacking PRs or branching off feature branches.

## Commit Rules
- **Commit identity**: `git -c user.name="hlin99" -c user.email="tony.lin@intel.com" commit -s`
- Always use `tony.lin@intel.com` as commit email. Never use noreply address.
- Always include `Signed-off-by` trailer (`-s` flag) for DCO compliance.
- Never add `Co-authored-by` trailers.
- Follow conventional commits: `<type>: <short description>` (fix, feat, test, docs, refactor, chore, ci).

## Before Pushing
1. Run pre-commit: `pre-commit run --all-files`
2. Run lint: `ruff check xpyd tests`
3. Run tests: `pytest tests/ -q`
4. All three must pass locally before pushing.

## Merge Policy
- **Bots must NEVER merge a PR.** All merges done by human maintainer.
- **Bots must NEVER close a PR.** Only human maintainer closes.
- Non-negotiable. Do not call merge or close API under any circumstances.

## CI
- CI must be 100% green before merge. No skips allowed.
- No test may be skipped. If a test can't run, fix it or remove it.

## Testing
- Unit tests in `tests/` — pure bench logic, no external dependencies.
- Integration tests in [xPyD-integration](https://github.com/xPyD-hub/xPyD-integration).

## Secrets
- Never hardcode tokens or credentials in code, PR descriptions, or bot prompts.

## Freshness
- **Always pull latest main and re-read BOT_POLICY.md before starting any work.** This is a living document. Never rely on cached copies.

## Architecture
- Lightweight PD-disaggregated LLM serving proxy.
- Follow vLLM bench CLI compatibility (see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md)).
