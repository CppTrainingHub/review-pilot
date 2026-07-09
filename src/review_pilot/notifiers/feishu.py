from __future__ import annotations

import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from review_pilot.notifiers.base import NotificationMessage, NotificationResult
from review_pilot.report_models import Finding, ReviewReport
from review_pilot.report_summary import SEVERITY_ORDER


class FeishuNotifierError(RuntimeError):
    pass


class HttpTransport(Protocol):
    def __call__(self, request: Request, timeout: float) -> Any:
        raise NotImplementedError


class FeishuNotifier:
    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        transport: HttpTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.webhook_url = webhook_url if webhook_url is not None else os.environ.get("FEISHU_WEBHOOK_URL")
        self.transport = transport or urlopen
        self.timeout = timeout

    def notify(
        self,
        message: NotificationMessage,
        *,
        dry_run: bool = False,
    ) -> NotificationResult:
        if dry_run:
            return NotificationResult(
                channel=message.channel,
                mode="feishu-dry-run",
                delivered=False,
                payload={
                    "webhook": "configured" if self.webhook_url else "missing",
                    **message.payload,
                },
            )

        if not self.webhook_url:
            raise FeishuNotifierError("FEISHU_WEBHOOK_URL is required when dry_run is false")

        request = Request(
            self.webhook_url,
            data=json.dumps(message.payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise FeishuNotifierError(f"feishu webhook failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise FeishuNotifierError(f"feishu webhook request failed: {exc.reason}") from exc

        parsed: dict[str, Any] | None = None
        if response_body.strip():
            try:
                value = json.loads(response_body)
            except json.JSONDecodeError as exc:
                raise FeishuNotifierError("feishu webhook response is not valid JSON") from exc
            if isinstance(value, dict):
                parsed = value

        return NotificationResult(
            channel=message.channel,
            mode="feishu",
            delivered=True,
            payload=message.payload,
            response=parsed,
        )


def message_from_report(
    report: ReviewReport,
    *,
    report_url: str | None = None,
) -> NotificationMessage:
    card = build_feishu_card(report, report_url=report_url)
    summary = _card_title(report)
    return NotificationMessage(
        channel="feishu",
        payload={
            "msg_type": "interactive",
            "card": card,
        },
        summary=summary,
    )


def build_feishu_card(
    report: ReviewReport,
    *,
    report_url: str | None = None,
) -> dict[str, Any]:
    summary = report.summary
    repo_info = report.repo_info or {}
    repository = str(repo_info.get("repository") or repo_info.get("root") or "unknown")
    pull_request = repo_info.get("pull_request")
    highest = str(summary.get("highest_severity") or "none")
    total = int(summary.get("total_findings") or 0)
    risk = _risk_label(highest, total)

    elements: list[dict[str, Any]] = [
        _markdown(
            f"**项目**：{repository}\n"
            f"**PR**：{('#' + str(pull_request)) if pull_request else 'unknown'}\n"
            f"**风险等级**：{risk}\n"
            f"**Finding 数量**：{total}\n"
            f"**最高风险**：{highest}"
        ),
        _markdown(_severity_line(summary)),
    ]
    top_findings = _top_findings(report.findings)
    if top_findings:
        elements.append(_markdown(_top_findings_line(top_findings)))
    if report_url:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "查看完整报告"},
                        "type": "primary",
                        "url": report_url,
                    }
                ],
            }
        )
    else:
        elements.append(_markdown("完整报告链接：未配置"))

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": _card_title(report)},
            "template": _risk_template(risk),
        },
        "elements": elements,
    }


def _markdown(content: str) -> dict[str, Any]:
    return {"tag": "markdown", "content": content}


def _severity_line(summary: dict[str, Any]) -> str:
    counts = summary.get("severity_counts")
    if not isinstance(counts, dict):
        counts = {}
    parts = [f"{severity}: {int(counts.get(severity, 0))}" for severity in SEVERITY_ORDER]
    return "**Severity**：" + " / ".join(parts)


def _top_findings_line(findings: list[Finding]) -> str:
    lines = ["**Top Findings**"]
    for finding in findings:
        lines.append(f"- [{finding.severity}] {_format_location(finding)} {finding.message}")
    return "\n".join(lines)


def _card_title(report: ReviewReport) -> str:
    summary = report.summary
    total = int(summary.get("total_findings") or 0)
    highest = summary.get("highest_severity") or "none"
    return f"Review Pilot 摘要：{total} 个 finding，最高 {highest}"


def _risk_label(highest: str, total: int) -> str:
    if total == 0:
        return "clean"
    if highest in {"P0", "P1"}:
        return "high"
    if highest == "P2":
        return "medium"
    return "low"


def _risk_template(risk: str) -> str:
    return {
        "high": "red",
        "medium": "orange",
        "low": "blue",
        "clean": "green",
    }.get(risk, "blue")


def _top_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda finding: finding.severity_rank)[:3]


def _format_location(finding: Finding) -> str:
    if finding.file_path and finding.line_no:
        return f"{finding.file_path}:{finding.line_no}"
    if finding.file_path:
        return finding.file_path
    return "unknown"
