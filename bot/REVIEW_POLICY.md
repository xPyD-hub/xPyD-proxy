<!-- CRITICAL: DO NOT SUMMARIZE OR COMPRESS THIS FILE -->
<!-- This file contains precise rules that must be read in full. -->

# Review Policy — xPyD-proxy

## Roles
| Role | GitHub Account | Action |
|------|---------------|--------|
| Implementer | `hlin99` | Write code, submit PRs |
| Reviewer 1 | `hlin99-Review-Bot` | Review PRs |
| Reviewer 2 | `hlin99-Review-BotX` | Review PRs |

Each reviewer uses its own dedicated token. Never use author's token for reviews.

## Reviewer Schedule
| Condition | Check Frequency |
|-----------|----------------|
| Open PRs exist | every 5 minutes |
| No open PRs | every 15 minutes |

## What to Review
1. Skip draft PRs.
2. Skip already-reviewed commits (only APPROVE counts as reviewed).
3. Re-requested reviews take priority — always perform fresh review.
4. One review per PR per commit SHA — never submit multiple reviews for same commit.

## Review Process

Review every non-draft PR with proxy-level strict standards. Every line examined.

## Review Checklist
For each non-draft PR with a new commit:

| Area | Check |
|---|---|
| CI | Must be fully green before APPROVE. May submit REQUEST_CHANGES or COMMENT while pending. |
| Merge conflicts | If mergeable == false, REQUEST_CHANGES. |
| Logic errors | Incorrect conditions, off-by-one, unhandled edge cases. |
| Type safety | Mismatched types, missing None checks. |
| Concurrency | Race conditions, missing locks, shared mutable state. |
| Exception handling | Bare except, swallowed exceptions, resource leaks. |
| Security | Injection risks, hardcoded secrets, unsanitized input. |
| Code style | Unused imports, shadowed variables, unclear naming. |
| Test coverage | New logic must have corresponding tests. |
| Design conformance | Implementation must match the linked GitHub Issue design. |

## Verdicts
- **APPROVE** — code correct, CI green, no issues.
- **REQUEST_CHANGES** — any issue found. Use inline comments.
- **COMMENT** — CI pending or noting something without blocking.

## Merge Policy
- **Bots must NEVER merge a PR.** Human maintainer only.
- **Bots must NEVER close a PR.** Human maintainer only.
- At least 1 approval required before human can merge.
