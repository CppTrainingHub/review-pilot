# Quickstart

这份 quickstart 用来跑通 review-pilot 的最终形态：本地 review、GitHub Actions artifact、PR Summary Comment 和飞书摘要通知。

## 安装

```bash
python -m pip install -e ".[dev]"
review-pilot doctor
```

## 本地 review

在任意 Git 仓库里制造一处 staged change，然后运行：

```bash
export REVIEW_PILOT_LLM_PROVIDER=openai-compatible
export REVIEW_PILOT_LLM_MODEL=deepseek-v4-pro
export REVIEW_PILOT_LLM_BASE_URL=https://api.deepseek.com
export REVIEW_PILOT_API_KEY='<your-deepseek-key>'
export REVIEW_PILOT_LLM_TIMEOUT_SECONDS=120
review-pilot review --staged --provider openai-compatible --format markdown --output review-report.md
```

## GitHub Actions artifact

仓库需要配置 Secret：`REVIEW_PILOT_API_KEY`。工作流触发后，`review-pilot github-action` 会读取真实 PR event，准备 PR 工作区，调用 DeepSeek，并生成 artifact。

artifact 里会生成：

```bash
review-pilot-artifacts/review-report.md
review-pilot-artifacts/review-report.json
```

## PR Summary Comment

workflow 需要 `contents: read`、`pull-requests: read`、`issues: write` 权限。summary comment 带有 `<!-- review-pilot-summary -->` marker，重复运行会更新旧评论。

## 飞书通知

真实发送时，把 webhook 放到环境变量：

```bash
export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/xxxx'
review-pilot notify feishu --report review-pilot-artifacts/review-report.json
```

在 GitHub Actions 里可以把 `FEISHU_WEBHOOK_URL` 放进仓库 Secret，然后在 artifact 生成后发送卡片：

```bash
review-pilot notify feishu --report review-pilot-artifacts/review-report.json
```

webhook 不应该写进仓库、截图、日志或教程正文。

## 全量测试

```bash
python -m pytest
```
