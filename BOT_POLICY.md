# BOT_POLICY.md — Automated Bot Rules

This file defines the rules that automated bots (CI review bots, cron-driven
agents, etc.) **must** follow when operating on this repository. Human
contributors should refer to `CONTRIBUTING.md`.

---

## When Acting as Reviewer

### Identity

Two review bots operate on this repo:

- **hlin99-Review-Bot** — first reviewer
- **hlin99-Review-BotX** — second reviewer

Each bot uses its own dedicated token. **Never use the author's (hlin99)
token for reviews.** The author's token is for authoring PRs only.

### What to Review

1. **Skip draft PRs** — do not review, comment, or interact with them.
2. **Skip already-reviewed commits** — if the PR head SHA has not changed since
   your last review, do not submit a duplicate review.
3. **One review per PR per commit SHA** — never submit multiple reviews for the
   same commit. If you submitted `CHANGES_REQUESTED` on a commit, do **not**
   submit `APPROVE` on the same commit — wait for the author to push a new
   commit that addresses the feedback, then re-review the new commit.

### Review Standard

Reviews must be performed to the **strictest standard**. Every line of changed
code must be examined. Do not approve unless you are confident the code is
correct.

**Design conformance is mandatory.** Before reviewing any PR, pull the latest
`main` and read `tasklist_openclaw.md` to understand the current task design.
Verify that the implementation matches the spec (YAML schema, API contract,
directory structure, etc.). If the code deviates from the design in
`tasklist_openclaw.md`, submit `REQUEST_CHANGES` — even if the code itself is
technically correct. An implementation that doesn't match the agreed design is
wrong.

### Review Checklist

For each non-draft PR with a new commit:

| Area | Check |
|---|---|
| **CI** | CI status does **not** block reviewing — start reviewing immediately. However, CI must be fully green before submitting `APPROVE`. If CI is pending or failed, you may still submit `REQUEST_CHANGES` or `COMMENT`. |
| **Merge conflicts** | If `mergeable == false`, submit `REQUEST_CHANGES`. |
| **`core/` changes** | Business logic and API signatures must remain intact. Topology matrix configs `(1,2,1) (2,2,1) (1,2,2) (1,2,4) (1,2,8) (2,2,2) (2,4,1) (2,4,2)` must not be broken. |
| **Logic errors** | Incorrect conditions, off-by-one, unhandled edge cases. |
| **Type safety** | Mismatched parameter/return types, missing `None` checks. |
| **Concurrency** | Race conditions, missing locks, shared mutable state. |
| **Exception handling** | Bare `except`, swallowed exceptions, resource leaks. |
| **Security** | Injection risks, hardcoded secrets, unsanitized input. |
| **Code style** | Unused imports, shadowed variables, unclear naming. |
| **Test coverage** | New logic must have corresponding tests. |

### Review Verdicts

- **`APPROVE`** — only if the code is correct, CI is fully green, and no issues
  are found.
- **`REQUEST_CHANGES`** — if any issue is found. Use inline comments to point
  to specific files and lines. The PR remains blocked until the author addresses
  all requested changes and pushes a new commit. Only then should the bot
  re-review and potentially approve.
- **`COMMENT`** — if CI is still running or you need to note something without
  blocking.

### Merge Policy

> **Bots must NEVER merge a PR.** All merge operations are performed manually
> by a human maintainer.

This is non-negotiable. Do not call the merge API under any circumstances.

**CI is a merge gate, not a review gate.** A PR cannot be merged until all CI
checks pass, but reviewers should not wait for CI to start reviewing code.

### Review Trigger Schedule

- **Has open (non-draft) PRs**: check every **5 minutes** for new commits and
  review comments.
- **No open PRs** (all draft or none): check every **15 minutes** to see if
  any draft PR has been marked ready for review or a new PR has been opened.
- When a new PR or new commit is detected, reset to the **5-minute** interval.
- A review can also be triggered immediately via chat command.

### Rate Limiting

- Respect GitHub API rate limits; back off on `429` responses.
- Do not flood PRs with duplicate comments or reviews.

---

## When Acting as Author (Opening PRs)

### Identity

Bot-authored PRs use the **hlin99** token (the repo owner account).

### Branch Rules

- **Never push directly to `main`.** All changes go through a PR.
- Branch from the latest `main`. Keep the branch up-to-date by merging `main`
  into it (not rebasing).
- **Each PR must be independent** — based on the latest `main`, with no
  dependencies between PRs. Do not stack PRs or branch off other feature
  branches.
- **Avoid force-push.** Always push new commits. Force-push destroys review
  history and is only acceptable when a maintainer explicitly requests it.
- Use descriptive branch names: `fix/issue-12-error-handling`,
  `feat/add-metrics`, `test/concurrent-edge-cases`.

### Before Pushing

1. **Run pre-commit hooks** — the repo uses pre-commit; run
   `pre-commit run --all-files`.
2. **Run the full test suite** — `python -m pytest tests/ -v`.
3. **Run linters** — `ruff check .` and `isort --check-only --skip core .`.
4. All three must pass locally before pushing.

### Commit Messages

Follow conventional commits:

```
<type>: <short description>

[optional body]
[optional footer]
```

Types: `fix`, `feat`, `test`, `docs`, `refactor`, `chore`, `ci`.

### Commit Identity

All commits must use the following identity:
```
git -c user.name="hlin99" -c user.email="tony.lin@intel.com" commit
```

Rules:
- Always use `tony.lin@intel.com` as the commit email
- Never use the GitHub noreply address (`*@users.noreply.github.com`)
- Never add `Co-authored-by` trailers to commit messages
- Never add `Signed-off-by` or other identity trailers

### PR Description

- Clearly state **what** changed and **why**.
- Reference related issues (e.g. `closes #12`).
- If modifying `core/`, explicitly call it out and explain the necessity.

### Responding to Reviews

- Address all `REQUEST_CHANGES` feedback before requesting re-review.
- Always push new commits to address feedback — do not amend or force-push.
- **Reply to each addressed review comment** with a reference to the fix
  commit (e.g. "Fixed in `abc1234`."). This makes it easy to distinguish
  resolved comments from new ones.
- **Re-request review** after pushing fixes — use the GitHub API or UI to
  re-request from the reviewer(s) who requested changes. Do not wait for
  the reviewer to notice the new commit on their own.
- Keep PRs focused — one concern per PR.

> These rules apply to **everyone** — bots and humans alike.

### Active PR Maintenance

The author bot runs a maintenance cron that triggers every **5 minutes** when
there are open (non-draft) PRs authored by the bot. On each trigger it must:

1. **Update branch** — if the PR branch is behind `main`, update it (merge
   `main` into the branch). PRs must always be up-to-date with `main`.
2. **CI check** — check CI status on the PR. If any check has failed, examine
   the failure logs, fix the code, and push a new commit. CI must be fully
   green. Do not wait for reviewers to point out CI failures — fix them
   proactively.
3. **Review comment check** — read any new `CHANGES_REQUESTED` reviews or
   inline comments. **Do not just check the latest review status** — even if
   a later reviewer submitted `APPROVE`, examine every `CHANGES_REQUESTED`
   review and verify that the specific code issues raised have been addressed
   by a subsequent commit. If a review points out a real code bug and no fix
   commit exists, you must fix the code. For each piece of feedback:
   - Fix the code accordingly.
   - Run pre-commit, tests, and linters locally before pushing.
   - Push a new commit (not amend/force-push over the reviewed commit).
   - **Reply to each review comment** that was addressed, referencing the
     fix commit (e.g. "Fixed in `abc1234`."). This makes it easy to see
     at a glance which comments have been resolved and which are new.
4. **Re-request review** — after pushing fixes, re-request review from the
   reviewer(s) who requested changes (via the GitHub API `POST
   /repos/{owner}/{repo}/pulls/{number}/requested_reviewers`).
5. **Repeat** — continue this cycle until the PR is approved or closed.

**No force-push.** Force-pushing destroys review context and makes it
impossible for reviewers to see incremental changes. Always push new commits.
The only acceptable exception is when a maintainer explicitly requests it.

When there are no open bot-authored PRs, the maintenance cron does not need to
run.

---

## General

- **Always fetch latest before acting** — `BOT_POLICY.md` and
  `tasklist_openclaw.md` are living documents that change frequently. Before
  starting any work (review, authoring, or maintenance), **pull the latest
  `main`** and re-read both files. Never rely on a cached or local copy.
  Implementing against an outdated design or policy is considered a bug.
- **English only** — all content on GitHub must be in English. This includes
  code, comments, commit messages, PR titles/descriptions, review comments,
  and inline annotations. No Chinese characters allowed anywhere in the repo
  or on GitHub.
- **Secrets** — never hardcode tokens or credentials in code, PR descriptions,
  or bot prompts. Read from secure storage at runtime.
- **Scope** — bots should limit their actions to reviewing code. No issue
  triage, no label management, no branch deletion unless explicitly configured.
- **Transparency** — every bot action should produce a brief summary of what it
  did (or chose not to do) for audit purposes.
