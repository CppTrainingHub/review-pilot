from __future__ import annotations

import io
import json
from urllib.request import Request

from review_pilot.cli import main
from review_pilot.notifiers.feishu import FeishuNotifier, build_feishu_card, message_from_report
from review_pilot.report_models import Finding, ReviewReport
from review_pilot.report_writer import write_report


class FakeResponse:
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return b'{"StatusCode": 0}'


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.body: dict[str, object] | None = None

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        assert request.data is not None
        self.body = json.loads(request.data.decode("utf-8"))
        return FakeResponse()


def run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(args, stdout=stdout, stderr=stderr)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_build_feishu_card_contains_summary_fields() -> None:
    card = build_feishu_card(_report(), report_url="https://ci.example/report")

    assert card["header"]["title"]["content"] == "Review Pilot 摘要：1 个 finding，最高 P1"
    rendered = json.dumps(card, ensure_ascii=False)
    assert "项目" in rendered
    assert "octo-org/review-demo" in rendered
    assert "PR" in rendered
    assert "#42" in rendered
    assert "查看完整报告" in rendered


def test_feishu_notifier_dry_run_does_not_call_transport() -> None:
    transport = FakeTransport()
    notifier = FeishuNotifier(webhook_url=None, transport=transport)

    result = notifier.notify(message_from_report(_report()), dry_run=True)

    assert result.mode == "feishu-dry-run"
    assert result.delivered is False
    assert result.payload["webhook"] == "missing"
    assert transport.requests == []


def test_feishu_notifier_posts_payload_when_not_dry_run() -> None:
    transport = FakeTransport()
    notifier = FeishuNotifier(
        webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/test",
        transport=transport,
    )

    result = notifier.notify(message_from_report(_report()), dry_run=False)

    assert result.mode == "feishu"
    assert result.delivered is True
    assert len(transport.requests) == 1
    assert transport.body is not None
    assert transport.body["msg_type"] == "interactive"


def test_notify_feishu_cli_renders_dry_run_payload(tmp_path) -> None:
    report_path = tmp_path / "review-report.json"
    report_path.write_text(write_report(_report(), "json"), encoding="utf-8")

    exit_code, stdout, stderr = run_cli(
        [
            "notify",
            "feishu",
            "--report",
            str(report_path),
            "--dry-run",
            "--report-url",
            "https://ci.example/report",
        ]
    )

    payload = json.loads(stdout)
    assert exit_code == 0
    assert payload["mode"] == "feishu-dry-run"
    assert payload["payload"]["webhook"] in {"configured", "missing"}
    assert "Review Pilot 摘要" in json.dumps(payload, ensure_ascii=False)
    assert stderr == ""


def test_notify_help_does_not_require_dry_run() -> None:
    exit_code, stdout, stderr = run_cli(["notify", "--help"])

    assert exit_code == 0
    assert "review-pilot notify" in stdout
    assert "feishu" in stdout
    assert "required" not in stdout.lower()
    assert stderr == ""


def _report() -> ReviewReport:
    return ReviewReport(
        findings=[
            Finding(
                message="Debug print leaked into review path",
                file_path="src/app.py",
                line_no=12,
                severity="P1",
                category="bug",
                source="rule",
            )
        ],
        repo_info={
            "repository": "octo-org/review-demo",
            "pull_request": 42,
        },
        config_source="default",
    )
