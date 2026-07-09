# review-pilot

review-pilot is a command-line code review agent for local Git changes, GitHub pull requests, and GitLab merge requests. It combines deterministic rules, optional Semgrep checks, repository context selection, LLM review, evidence filtering, Markdown/JSON reports, PR/MR summary comments, and Feishu notifications.

## What Works Now

- Install the project in editable mode.
- Run `review-pilot --help`.
- Run `review-pilot --version`.
- Run `review-pilot doctor`.
- Run `review-pilot review --help`.
- Run the same CLI through `python -m review_pilot`.
- Run `review-pilot repo-info --json` inside a Git repository.
- Run `review-pilot diff --staged --raw` to print staged raw diff text.
- Parse unified diff files, hunks, line kinds, and old/new line numbers.
- Generate deterministic Markdown and JSON review reports.
- Run configurable rules, normalize findings, and apply CI failure thresholds.
- Select repository context, apply a token budget, and build an auditable Context Pack.
- Run registered tools safely, integrate Semgrep, and filter tool findings to changed lines.
- Install managed pre-commit and pre-push hooks.
- Run `review-pilot llm doctor` without exposing API keys.
- Preview the versioned prompt with `review-pilot prompt-preview --staged --provider fake`.
- Validate saved model output with `review-pilot llm validate-output --input <file>`.
- Validate saved findings against staged evidence with `review-pilot evidence-check --staged --input <file>`.
- Run `review-pilot review --staged --provider fake` and receive evidence-verified LLM findings plus dropped-finding statistics.
- Run `review-pilot review --staged --provider fake --format json` and receive a final merged JSON report with `merge_summary`.
- Run `review-pilot review --staged --provider fake --with-tools --output report.md` and write a final Markdown report that preserves rule, Semgrep, and LLM sources.
- Run `review-pilot review --staged` and receive a local end-to-end review report instead of raw diff text.
- Run `review-pilot review --staged --no-ai --with-tools --output report.md` for deterministic local review with tool status.
- Run `review-pilot review --staged --provider fake --profile pre-push --output report.md` to exercise the local profile path with context, tools, LLM evidence guard, merge, and report writing.
- Read a GitHub PR and prepare a remote workspace with `review-pilot review-pr <url> --dry-run --no-ai`.
- Run `review-pilot github-action --event-path <event.json> --output-dir review-pilot-artifacts --dry-run` to generate GitHub Actions artifacts.
- Publish or preview a PR summary comment with `review-pilot github-action --event-path <event.json> --post-summary-comment`.
- Run `review-pilot gitlab-ci --output-dir review-pilot-artifacts --provider openai-compatible --post-summary-comment` from a GitLab merge request pipeline.
- Render or send a Feishu summary card with `review-pilot notify feishu --report review-pilot-artifacts/review-report.json`.
- Run `review-pilot naive-review --staged --provider fake` for deterministic naive review output.
- Run `review-pilot naive-review --staged --provider openai` with an OpenAI-compatible API key.

## Quickstart

See `docs/quickstart.md`.

For a small local demo project, see `examples/final-demo`.

## Commands

```bash
python -m pip install -e '.[dev]'
review-pilot --help
review-pilot --version
review-pilot doctor
review-pilot review --help
review-pilot repo-info --json
review-pilot diff --staged --raw
review-pilot llm doctor
review-pilot prompt-preview --staged --provider fake
review-pilot llm validate-output --input tests/fixtures/llm/valid_findings.json
review-pilot evidence-check --staged --input tests/fixtures/llm/valid_findings.json
review-pilot review --staged --provider fake
review-pilot review --staged --provider fake --format json
review-pilot review --staged --provider fake --with-tools --output report.md
review-pilot review --staged --no-ai --with-tools --output report.md
review-pilot review --staged --provider fake --profile pre-push --output report.md
review-pilot review-pr https://github.com/OWNER/REPO/pull/123 --dry-run --no-ai
review-pilot github-action --event-path tests/fixtures/github/pull_request_event.json --output-dir review-pilot-artifacts --dry-run
review-pilot github-action --event-path tests/fixtures/github/pull_request_event.json --output-dir review-pilot-artifacts --dry-run --post-summary-comment
review-pilot gitlab-ci --output-dir review-pilot-artifacts --dry-run
review-pilot notify feishu --report review-pilot-artifacts/review-report.json --dry-run
review-pilot naive-review --staged --provider fake
python -m review_pilot --help
python -m pytest
```
