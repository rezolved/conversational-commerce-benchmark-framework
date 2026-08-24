"""Output helpers — summary building, file writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_summary(
    report: dict[str, Any],
    model_tag: str,
    dialog_file: str,
    output_dir: Path,
) -> str:
    """Build a human-readable summary table from the benchmark report."""
    results = report["results"]
    successful = [r for r in results if r["status"] == "success"]

    all_scores: dict[str, list[float]] = {}
    judged_count = 0
    for r in successful:
        j = r.get("judgement")
        if not j:
            continue
        judged_count += 1
        if isinstance(j, dict):
            for key, val in j.items():
                if isinstance(val, (int, float)):
                    all_scores.setdefault(key, []).append(float(val))
                elif isinstance(val, dict) and "score" in val:
                    all_scores.setdefault(key, []).append(float(val["score"]))

    avg_latency = 0.0
    if successful:
        avg_latency = sum(r["total_latency_seconds"] for r in successful) / len(successful)

    lines = [
        "=" * 65,
        "  Conversational Commerce Benchmark — Run Summary",
        "=" * 65,
        f"  Model           : {model_tag}",
        f"  Dialog file     : {dialog_file}",
        f"  Total scenarios : {report['total']}",
        f"  Successful      : {report['success']}",
        f"  Failed          : {report['failed']}",
        f"  Judged          : {judged_count}",
        f"  Avg latency     : {avg_latency:.2f}s per conversation",
        "-" * 65,
    ]

    if all_scores:
        lines.append("  Judge Scores (averages):")
        for rubric, scores in sorted(all_scores.items()):
            avg = sum(scores) / len(scores)
            lines.append(f"    {rubric:30s} : {avg:.2f}  (n={len(scores)})")
    else:
        lines.append("  Judge Scores: (none available)")

    lines.extend(["-" * 65, f"  Outputs in      : {output_dir}", "=" * 65])
    return "\n".join(lines)


def write_model_outputs(
    full_output: dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist benchmark-report.json, judge-results.json, and summary.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = full_output["results"]
    model_tag = full_output["model_tag"]
    dialog_file = full_output.get("dialog_file", "unknown")

    report_path = output_dir / "benchmark-report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    judge_results = [
        {
            "index": r["index"],
            "scenario_id": r["scenario_id"],
            "judgement": r.get("judgement"),
            "judgement_error": r.get("judgement_error"),
        }
        for r in results
        if r["status"] == "success"
    ]
    judge_path = output_dir / "judge-results.json"
    with judge_path.open("w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2, ensure_ascii=False)

    report_for_summary = {
        "total": full_output.get("total_scenarios", len(results)),
        "success": full_output.get("successful", sum(1 for r in results if r["status"] == "success")),
        "failed": full_output.get("failed", sum(1 for r in results if r["status"] != "success")),
        "results": results,
    }
    summary = build_summary(report_for_summary, model_tag, dialog_file, output_dir)
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n{summary}")
