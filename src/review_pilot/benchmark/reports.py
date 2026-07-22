from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def render_benchmark_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    run_config = report.get("run_config", {})
    strategy = run_config.get("strategy", "baseline")
    lines = [
        f"# AACR-Bench {strategy} Report",
        "",
        f"- **Dataset:** {report.get('benchmark', 'AACR-Bench')}",
        f"- **Provider:** {report.get('run_config', {}).get('provider', 'unknown')}",
        f"- **Model:** {report.get('run_config', {}).get('model', 'unknown')}",
        f"- **Completed cases:** {summary.get('completed_cases', 0)}",
        f"- **Failed cases:** {summary.get('failed_cases', 0)}",
        "",
        "## Reproducibility",
        "",
        f"- **Dataset SHA-256:** `{report.get('dataset', {}).get('sha256', '')}`",
        f"- **Sample set SHA-256:** `{report.get('sample_set', {}).get('sha256') or 'none'}`",
        f"- **Prompt version:** `{report.get('run_config', {}).get('prompt_version', '')}`",
        f"- **LLM schema version:** `{report.get('run_config', {}).get('llm_schema_version', '')}`",
        f"- **Matcher:** `{report.get('run_config', {}).get('matcher', '')}`",
        f"- **Context budget:** `{report.get('run_config', {}).get('max_context_tokens', '')}` tokens",
    ]
    if strategy == "review-units":
        lines.append(
            f"- **ReviewUnit workers:** `{run_config.get('review_unit_workers', 1)}`"
        )
    lines.extend([
        "",
        "## Overall Metrics",
        "",
    ])
    lines.extend(_metric_table(summary))
    reflection = report.get("reflection", summary.get("reflection"))
    if isinstance(reflection, dict):
        lines.extend(_reflection_table(reflection))
    if report.get("run_config", {}).get("dynamic_context"):
        lines.extend(_context_recall_table(report.get("context_recall", {})))
    comparison = report.get("comparison")
    if isinstance(comparison, dict):
        lines.extend(_comparison_table(comparison))
        baseline_summary = comparison.get("baseline", {}).get("summary", {})
        review_units_summary = comparison.get("review-units", {}).get("summary", {})
        baseline_cases = baseline_summary.get("cases")
        review_units_cases = review_units_summary.get("cases")
        if (
            isinstance(baseline_cases, int)
            and isinstance(review_units_cases, int)
            and baseline_cases != review_units_cases
        ):
            lines.extend(
                [
                    "## Coverage Note",
                    "",
                    (
                        f"The baseline covers {baseline_cases} cases, while review-units "
                        f"covers {review_units_cases}; the strategy metrics are shown for "
                        "format and progress tracking and are not a full-sample comparison."
                    ),
                    "",
                ]
            )
    lines.extend(_group_table("By Language", report.get("by_language", {})))
    lines.extend(_group_table("By Category", report.get("by_category", {})))
    lines.extend(_group_table("By Context Level", report.get("by_context_level", {})))

    lines.extend(["## Cases", "", "| Case | Language | Status | Expected | Generated | Matches | F1 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for case in report.get("cases", []):
        metrics = case.get("metrics", {})
        lines.append(
            "| {case_id} | {language} | {status} | {expected} | {generated} | {matched} | {f1:.4f} |".format(
                case_id=case.get("case_id", ""),
                language=case.get("language", ""),
                status=case.get("status", ""),
                expected=metrics.get("expected_comments", 0),
                generated=metrics.get("generated_comments", 0),
                matched=metrics.get("positive_matches", 0),
                f1=float(metrics.get("f1", 0.0)),
            )
        )
        if case.get("error"):
            lines.append(f"  - Error: `{case['error']}`")
    lines.append("")
    return "\n".join(lines)


def _comparison_table(comparison: Mapping[str, Any]) -> list[str]:
    baseline = comparison.get("baseline", {})
    review_units = comparison.get("review-units", {})
    baseline_summary = baseline.get("summary", {}) if isinstance(baseline, dict) else {}
    unit_summary = review_units.get("summary", {}) if isinstance(review_units, dict) else {}
    lines = [
        "## Strategy Comparison",
        "",
        "| Metric | baseline | review-units |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("line_accuracy", "Line Accuracy"),
    ):
        lines.append(
            f"| {label} | {float(baseline_summary.get(key, 0.0)):.6f} | "
            f"{float(unit_summary.get(key, 0.0)):.6f} |"
        )
    lines.append("")
    return lines


def _metric_table(metrics: Mapping[str, Any]) -> list[str]:
    rows = [
        ("Precision", metrics.get("precision", 0.0)),
        ("Recall", metrics.get("recall", 0.0)),
        ("F1", metrics.get("f1", 0.0)),
        ("Line Accuracy", metrics.get("line_accuracy", 0.0)),
        ("Line Precision", metrics.get("line_precision", 0.0)),
        ("Line Recall", metrics.get("line_recall", 0.0)),
        ("Location Failure Rate", metrics.get("location_failure_rate")),
        ("Noise Rate", metrics.get("noise_rate", 0.0)),
        ("Average Latency (ms)", metrics.get("average_latency_ms", metrics.get("latency_ms", 0.0))),
        ("Token Cost (USD)", metrics.get("token_cost_usd")),
        ("Average Comments", metrics.get("average_comments", 0.0)),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for name, value in rows:
        if isinstance(value, float):
            rendered = f"{value:.6f}" if "Latency" not in name else f"{value:.3f}"
        else:
            rendered = "unknown" if value is None else str(value)
        lines.append(f"| {name} | {rendered} |")
    lines.append("")
    return lines


def _reflection_table(reflection: Mapping[str, Any]) -> list[str]:
    usage = reflection.get("token_usage", {})
    total_tokens = usage.get("total_tokens", 0) if isinstance(usage, Mapping) else 0
    lines = [
        "## Reflection",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Enabled | {str(bool(reflection.get('enabled', False))).lower()} |",
        f"| Eligible | {reflection.get('eligible', 0)} |",
        f"| Reviewed | {reflection.get('reviewed', 0)} |",
        f"| Keep | {reflection.get('keep', 0)} |",
        f"| Downgrade | {reflection.get('downgrade', 0)} |",
        f"| Drop | {reflection.get('drop', 0)} |",
        f"| Errors | {reflection.get('errors', 0)} |",
        f"| Reflection Tokens | {total_tokens} |",
        "",
    ]
    return lines


def _context_recall_table(groups: Any) -> list[str]:
    if not isinstance(groups, dict):
        return []
    lines = [
        "## Recall by Context Level",
        "",
        "| Context Level | Cases | Recall | F1 | tool_calls | dynamic_tokens | token_cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in sorted(groups.items()):
        label = " ".join(part.capitalize() for part in str(name).split())
        token_cost = metrics.get("token_cost_usd")
        rendered_cost = "unknown" if token_cost is None else f"{float(token_cost):.8f}"
        lines.append(
            f"| {label} | {metrics.get('cases', 0)} | "
            f"{float(metrics.get('recall', 0.0)):.6f} | "
            f"{float(metrics.get('f1', 0.0)):.6f} | "
            f"{metrics.get('tool_calls', 0)} | {metrics.get('dynamic_tokens', 0)} | "
            f"{rendered_cost} |"
        )
    lines.append("")
    return lines


def _group_table(title: str, groups: Any) -> list[str]:
    if not isinstance(groups, dict):
        return []
    lines = [f"## {title}", "", "| Group | Cases | Expected | Generated | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in sorted(groups.items()):
        lines.append(
            "| {name} | {cases} | {expected} | {generated} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                name=name,
                cases=metrics.get("cases", 0),
                expected=metrics.get("expected_comments", 0),
                generated=metrics.get("generated_comments", 0),
                precision=float(metrics.get("precision", 0.0)),
                recall=float(metrics.get("recall", 0.0)),
                f1=float(metrics.get("f1", 0.0)),
            )
        )
    lines.append("")
    return lines


def write_benchmark_reports(
    report: Mapping[str, Any],
    output_dir: str,
) -> tuple[str, str]:
    from pathlib import Path

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "report.json"
    markdown_path = target / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_benchmark_markdown(report),
        encoding="utf-8",
    )
    return str(json_path), str(markdown_path)


REQUIRED_COMPARISON_GROUPS = (
    "baseline-30",
    "review-units",
    "dynamic-context",
    "snippet-location",
    "reflection",
)


def compare_benchmark_reports(
    result_dirs: list[str],
    output_path: str,
) -> dict[str, Any]:
    reports: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    for directory in result_dirs:
        path = Path(directory)
        report_path = path / "report.json" if path.is_dir() else path
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read benchmark report {report_path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
            raise ValueError(f"invalid benchmark report: {report_path}")
        name = _comparison_name(path)
        if name in reports:
            raise ValueError(f"duplicate comparison group: {name}")
        reports[name] = payload
        paths[name] = report_path

    missing = [name for name in REQUIRED_COMPARISON_GROUPS if name not in reports]
    if missing:
        raise ValueError(
            "benchmark compare requires groups: "
            + ", ".join(REQUIRED_COMPARISON_GROUPS)
            + f"; missing: {', '.join(missing)}"
        )

    groups = {
        name: {
            "report_path": str(paths[name]),
            "summary": dict(reports[name]["summary"]),
            "run_config": dict(reports[name].get("run_config", {})),
        }
        for name in REQUIRED_COMPARISON_GROUPS
    }
    _validate_comparison_compatibility(reports)
    payload = {
        "schema_version": "review-pilot.aacr-final-comparison.v1",
        "benchmark": "AACR-Bench",
        "groups": groups,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_final_comparison(payload), encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _validate_comparison_compatibility(
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    fields = (
        ("dataset.sha256", lambda report: report.get("dataset", {}).get("sha256")),
        ("sample_set.sha256", lambda report: report.get("sample_set", {}).get("sha256")),
        ("run_config.provider", lambda report: report.get("run_config", {}).get("provider")),
        ("run_config.model", lambda report: report.get("run_config", {}).get("model")),
        ("run_config.prompt_version", lambda report: report.get("run_config", {}).get("prompt_version")),
        ("run_config.matcher", lambda report: report.get("run_config", {}).get("matcher")),
    )
    for field_name, getter in fields:
        values = {getter(report) for report in reports.values() if getter(report)}
        if len(values) > 1:
            raise ValueError(
                f"benchmark compare requires matching {field_name}; got {sorted(values)}"
            )


def _comparison_name(path: Path) -> str:
    return {
        "baseline-30-real": "baseline-30",
        "baseline-30": "baseline-30",
        "review-units": "review-units",
        "dynamic-context-30-real": "dynamic-context",
        "dynamic-context": "dynamic-context",
        "snippet-location-30-real": "snippet-location",
        "snippet-location": "snippet-location",
        "reflection": "reflection",
    }.get(path.name, path.name)


def _render_final_comparison(payload: Mapping[str, Any]) -> str:
    groups = payload["groups"]
    lines = [
        "# AACR-Bench Final Comparison",
        "",
        "| Group | Cases | Precision | Recall | F1 | Noise Rate | Average Latency (ms) | Token Cost (USD) | Reflection Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in REQUIRED_COMPARISON_GROUPS:
        summary = groups[name]["summary"]
        token_cost = summary.get("token_cost_usd")
        reflection = summary.get("reflection", {})
        reflection_usage = reflection.get("token_usage", {}) if isinstance(reflection, dict) else {}
        reflection_tokens = reflection_usage.get("total_tokens", 0) if isinstance(reflection_usage, dict) else 0
        lines.append(
            "| {name} | {cases} | {precision:.6f} | {recall:.6f} | {f1:.6f} | "
            "{noise:.6f} | {latency:.3f} | {cost} | {reflection_tokens} |".format(
                name=name,
                cases=summary.get("completed_cases", summary.get("cases", 0)),
                precision=float(summary.get("precision", 0.0)),
                recall=float(summary.get("recall", 0.0)),
                f1=float(summary.get("f1", 0.0)),
                noise=float(summary.get("noise_rate", 0.0)),
                latency=float(summary.get("average_latency_ms", 0.0)),
                cost=("unknown" if token_cost is None else f"{float(token_cost):.8f}"),
                reflection_tokens=reflection_tokens,
            )
        )
    lines.extend(
        [
            "",
            "## Reading the comparison",
            "",
            "All five rows must use the same dataset hash, sample-set hash, provider, model, and matcher version.",
            "Precision, Recall, F1, Noise Rate, latency, and token cost should be read together.",
            "",
        ]
    )
    return "\n".join(lines)
