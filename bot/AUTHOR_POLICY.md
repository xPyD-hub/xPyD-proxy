<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Author Policy — xPyD-proxy

Rules for the bot that writes code and submits PRs.

## Identity
| Role | GitHub Account |
|------|---------------|
| Author | `hlin99` |

## Before Coding
1. Pull latest main: `git pull origin main`
2. Create branch: `git checkout -b <type>/<short-description>`
3. Read [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) for architecture constraints.

## Code Quality
- Run pre-commit: `pre-commit run --all-files`
- Run lint: `ruff check xpyd tests`
- Run tests: `pytest tests/ -q`
- All must pass locally before pushing.

## PR Submission
1. One PR per task. Don't bundle unrelated changes.
2. Descriptive title: `type: short description`
3. PR body: what changed, why, test coverage, breaking changes. Reference issues (`closes #N`).
4. All CI must pass before requesting review.

## Responding to Review
1. Address ALL `REQUEST_CHANGES` feedback before requesting re-review.
2. Always push new commits — never amend or force-push.
3. Reply to each review comment with fix commit ref (e.g. "Fixed in `abc1234`").
4. Re-request review after pushing fixes (don't wait for reviewer to notice).
5. If reviewer is wrong, explain with evidence (link to source, docs, spec).


## Documentation Updates
Every PR must update relevant documentation:

| Change Type | Update |
|---|---|
| New feature / CLI argument | `docs/guide.md` |
| Architecture change | `docs/architecture.md` |
| Design decision | `docs/design.md` |
| Quick Start affected | `README.md` (keep one screen, link to guide.md) |

