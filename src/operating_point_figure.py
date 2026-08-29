from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Raised when committed model evidence is missing or inconsistent."""


FLOAT_TOLERANCE = 1e-9
REQUIRED_METRIC_FIELDS = {
    "model", "split", "tp", "fp", "tn", "fn", "precision", "recall", "accuracy",
    "flagged_rate", "false_positive_rate", "average_precision", "roc_auc", "threshold", "alerts",
}


@dataclass(frozen=True)
class OperatingPointEvidence:
    project_root: Path
    model_name: str
    threshold: float
    validation: dict[str, float]
    test: dict[str, float]
    validation_curve: tuple[dict[str, float], ...]
    validation_rows: int
    validation_frauds: int
    test_rows: int
    test_frauds: int
    validation_prevalence: float
    near_optimal: dict[str, float]


def _fail(message: str) -> None:
    raise EvidenceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"Missing required evidence file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Could not read JSON evidence at {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"Expected an object in {path}")
    return value


def _read_text(path: Path) -> str:
    if not path.is_file():
        _fail(f"Missing required evidence file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"Could not read evidence at {path}: {exc}")


def _number(value: Any, label: str, *, integer: bool = False) -> float | int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} is not numeric: {value!r}")
        raise AssertionError from exc
    if not math.isfinite(parsed):
        _fail(f"{label} is not finite: {value!r}")
    if integer:
        if not parsed.is_integer():
            _fail(f"{label} must be an integer: {value!r}")
        return int(parsed)
    return parsed


def _probability(value: Any, label: str) -> float:
    parsed = float(_number(value, label))
    if not 0.0 <= parsed <= 1.0:
        _fail(f"{label} must be between 0 and 1: {parsed}")
    return parsed


def _close(left: float, right: float, *, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _require_close(left: float, right: float, label: str) -> None:
    if not _close(left, right):
        _fail(f"{label} is inconsistent: {left} versus {right}")


def _metric_record(row: dict[str, str], label: str) -> dict[str, float]:
    record: dict[str, float] = {}
    for field in ("tp", "fp", "tn", "fn", "alerts"):
        record[field] = float(_number(row.get(field), f"{label}.{field}", integer=True))
    for field in (
        "precision", "recall", "accuracy", "flagged_rate", "false_positive_rate", "average_precision", "roc_auc", "threshold"
    ):
        record[field] = _probability(row.get(field), f"{label}.{field}")
    for field in ("missed_fraud_cost", "expected_cost", "expected_cost_per_transaction", "cost_savings_vs_no_model"):
        if row.get(field) not in (None, ""):
            record[field] = float(_number(row[field], f"{label}.{field}"))
    return record


def _validate_metric_record(record: dict[str, float], label: str) -> tuple[int, int]:
    tp = int(record["tp"])
    fp = int(record["fp"])
    tn = int(record["tn"])
    fn = int(record["fn"])
    rows = tp + fp + tn + fn
    frauds = tp + fn
    if rows <= 0 or frauds <= 0 or rows == frauds:
        _fail(f"{label} has unusable confusion counts: {record}")
    _require_close(record["precision"], tp / (tp + fp) if tp + fp else 0.0, f"{label}.precision")
    _require_close(record["recall"], tp / frauds, f"{label}.recall")
    _require_close(record["accuracy"], (tp + tn) / rows, f"{label}.accuracy")
    _require_close(record["flagged_rate"], (tp + fp) / rows, f"{label}.flagged_rate")
    _require_close(record["false_positive_rate"], fp / (fp + tn) if fp + tn else 0.0, f"{label}.false_positive_rate")
    if int(record["alerts"]) != tp + fp:
        _fail(f"{label}.alerts does not match tp + fp")
    return rows, frauds


def _read_metrics(path: Path, model_name: str) -> dict[str, dict[str, float]]:
    if not path.is_file():
        _fail(f"Missing required evidence file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_METRIC_FIELDS.difference(headers))
            if missing:
                _fail(f"Metrics summary is missing columns: {missing}")
            selected = [row for row in reader if row.get("model") == model_name]
    except OSError as exc:
        _fail(f"Could not read metrics summary at {path}: {exc}")
    split_labels = {row.get("split") for row in selected}
    if split_labels != {"validation", "test"}:
        _fail(f"Expected explicit validation and test rows for {model_name}; found {sorted(split_labels)}")
    records: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        rows = [row for row in selected if row.get("split") == split]
        if len(rows) != 1:
            _fail(f"Expected exactly one {split} row for {model_name}; found {len(rows)}")
        records[split] = _metric_record(rows[0], f"metrics_summary.{model_name}.{split}")
        _validate_metric_record(records[split], f"metrics_summary.{model_name}.{split}")
    return records


def _read_class_distribution(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        _fail(f"Missing required evidence file: {path}")
    result: dict[str, dict[str, int]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "rows", "legitimate", "fraud", "fraud_rate"}
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            if missing:
                _fail(f"Class distribution is missing columns: {missing}")
            for row in reader:
                split = row.get("split", "")
                if split in ("validation", "test"):
                    result[split] = {
                        "rows": int(_number(row.get("rows"), f"class_distribution.{split}.rows", integer=True)),
                        "legitimate": int(_number(row.get("legitimate"), f"class_distribution.{split}.legitimate", integer=True)),
                        "fraud": int(_number(row.get("fraud"), f"class_distribution.{split}.fraud", integer=True)),
                    }
    except OSError as exc:
        _fail(f"Could not read class distribution at {path}: {exc}")
    if set(result) != {"validation", "test"}:
        _fail(f"Class distribution must contain validation and test rows; found {sorted(result)}")
    for split, record in result.items():
        if record["rows"] != record["legitimate"] + record["fraud"]:
            _fail(f"class_distribution.{split} does not reconcile rows and classes")
    return result


def _read_curve(path: Path) -> tuple[dict[str, float], ...]:
    if not path.is_file():
        _fail(f"Missing required evidence file: {path}")
    required = {"threshold", "precision", "recall", "tp", "fp"}
    curve: list[dict[str, float]] = []
    previous_threshold: float | None = None
    previous_recall: float | None = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            if missing:
                _fail(f"Validation precision-recall curve is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                label = f"validation_curve.line_{line_number}"
                threshold = _probability(row.get("threshold"), f"{label}.threshold")
                precision = _probability(row.get("precision"), f"{label}.precision")
                recall = _probability(row.get("recall"), f"{label}.recall")
                tp = float(_number(row.get("tp"), f"{label}.tp", integer=True))
                fp = float(_number(row.get("fp"), f"{label}.fp", integer=True))
                if previous_threshold is not None and threshold > previous_threshold + FLOAT_TOLERANCE:
                    _fail("Validation curve thresholds must be non-increasing")
                if previous_recall is not None and recall + FLOAT_TOLERANCE < previous_recall:
                    _fail("Validation curve recall must be non-decreasing")
                curve.append({"threshold": threshold, "precision": precision, "recall": recall, "tp": tp, "fp": fp})
                previous_threshold = threshold
                previous_recall = recall
    except OSError as exc:
        _fail(f"Could not read validation curve at {path}: {exc}")
    if not curve:
        _fail("Validation precision-recall curve is empty")
    return tuple(curve)


def _compare_artifact_metrics(artifact_metrics: Any, summary_metrics: dict[str, float], split: str) -> None:
    if not isinstance(artifact_metrics, dict):
        _fail(f"Artifact metrics.{split} must be an object")
    for field in ("tp", "fp", "tn", "fn"):
        artifact_value = int(_number(artifact_metrics.get(field), f"artifact.metrics.{split}.{field}", integer=True))
        if artifact_value != int(summary_metrics[field]):
            _fail(f"Artifact and metrics summary disagree for {split}.{field}")
    for field in ("precision", "recall", "average_precision", "threshold"):
        artifact_value = _probability(artifact_metrics.get(field), f"artifact.metrics.{split}.{field}")
        _require_close(artifact_value, summary_metrics[field], f"artifact.metrics.{split}.{field}")


def load_evidence(project_root: Path) -> OperatingPointEvidence:
    project_root = Path(project_root).resolve()
    artifact = _read_json(project_root / "artifacts" / "fraud_model.json")
    model_name = artifact.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        _fail("artifacts/fraud_model.json has no selected model_name")
    threshold = _probability(artifact.get("threshold"), "artifact.threshold")

    report_text = _read_text(project_root / "reports" / "model_report.md")
    champion_match = re.search(r"Champion:\s+`([^`]+)`", report_text)
    if not champion_match or champion_match.group(1) != model_name:
        _fail("Model report champion does not match artifacts/fraud_model.json")
    report_lower = report_text.lower()
    for phrase in ("chronological", "test set is used only after", "illustrative", "not customer-facing reason codes"):
        if phrase not in report_lower:
            _fail(f"Model report is missing required boundary wording: {phrase}")
    run_status = _read_text(project_root / "RUN_STATUS.md").lower()
    if "near-tie" not in run_status or "test precision" not in run_status:
        _fail("RUN_STATUS.md is missing the model-selection/test-evaluation boundary")
    monitoring = _read_text(project_root / "MONITORING.md").lower()
    if "illustrative" not in monitoring or "ownership" not in monitoring:
        _fail("MONITORING.md is missing cost or ownership boundary wording")

    summary = _read_metrics(project_root / "reports" / "metrics_summary.csv", model_name)
    artifact_metrics = artifact.get("metrics")
    if not isinstance(artifact_metrics, dict):
        _fail("Artifact has no metrics object")
    for split in ("validation", "test"):
        _compare_artifact_metrics(artifact_metrics.get(split), summary[split], split)
    validation_rows, validation_frauds = _validate_metric_record(summary["validation"], "validation")
    test_rows, test_frauds = _validate_metric_record(summary["test"], "test")
    distribution = _read_class_distribution(project_root / "reports" / "class_distribution.csv")
    for split, rows, frauds in (("validation", validation_rows, validation_frauds), ("test", test_rows, test_frauds)):
        if distribution[split]["rows"] != rows or distribution[split]["fraud"] != frauds:
            _fail(f"Class distribution does not match exact {split} confusion support")

    curve = _read_curve(project_root / "reports" / "validation_precision_recall_curve.csv")
    last_curve = curve[-1]
    if int(last_curve["tp"] + last_curve["fp"]) != validation_rows:
        _fail("Validation curve support does not match validation confusion support")
    if int(last_curve["tp"]) != validation_frauds:
        _fail("Validation curve positive support does not match validation confusion support")
    selected_rows = [row for row in curve if _close(row["threshold"], threshold, tolerance=1e-12)]
    if len(selected_rows) != 1:
        _fail(f"Selected threshold {threshold:.12f} does not identify exactly one validation curve point")
    selected_curve = selected_rows[0]
    for field in ("precision", "recall"):
        _require_close(selected_curve[field], summary["validation"][field], f"validation_curve.selected.{field}")
    for field in ("tp", "fp"):
        if int(selected_curve[field]) != int(summary["validation"][field]):
            _fail(f"Selected curve point does not match validation {field}")

    threshold_selection = artifact.get("threshold_selection")
    if not isinstance(threshold_selection, dict):
        _fail("Artifact has no threshold_selection object")
    near_optimal: dict[str, float] = {}
    for field in (
        "near_optimal_threshold_min", "near_optimal_threshold_max", "near_optimal_precision_min", "near_optimal_precision_max",
        "near_optimal_recall_min", "near_optimal_recall_max",
    ):
        near_optimal[field] = _probability(threshold_selection.get(field), f"artifact.threshold_selection.{field}")
    if near_optimal["near_optimal_threshold_min"] > near_optimal["near_optimal_threshold_max"]:
        _fail("Near-optimal threshold band is reversed")
    if not near_optimal["near_optimal_threshold_min"] - FLOAT_TOLERANCE <= threshold <= near_optimal["near_optimal_threshold_max"] + FLOAT_TOLERANCE:
        _fail("Selected threshold is outside the committed near-optimal threshold band")

    return OperatingPointEvidence(
        project_root=project_root,
        model_name=model_name,
        threshold=threshold,
        validation=summary["validation"],
        test=summary["test"],
        validation_curve=curve,
        validation_rows=validation_rows,
        validation_frauds=validation_frauds,
        test_rows=test_rows,
        test_frauds=test_frauds,
        validation_prevalence=validation_frauds / validation_rows,
        near_optimal=near_optimal,
    )


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _text(x: float, y: float, value: str, *, size: int = 14, fill: str = "#263238", weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}" '
        f'font-family="system-ui, -apple-system, BlinkMacSystemFont, \'Segoe UI\', sans-serif" '
        f'font-size="{size}px" font-weight="{weight}" fill="{fill}">{_esc(value)}</text>'
    )


def _downsample_curve(curve: tuple[dict[str, float], ...], selected_index: int, max_points: int = 900) -> list[dict[str, float]]:
    if len(curve) <= max_points:
        return list(curve)
    indices = {0, len(curve) - 1, selected_index}
    for position in range(max_points):
        indices.add(int(round(position * (len(curve) - 1) / (max_points - 1))))
    ordered = sorted(indices)
    if len(ordered) > max_points:
        removable = [index for index in ordered if index not in {0, len(curve) - 1, selected_index}]
        ordered = sorted({0, len(curve) - 1, selected_index, *removable[: max_points - 3]})
    return [curve[index] for index in ordered]


def build_svg(evidence: OperatingPointEvidence) -> str:
    width, height = 1200, 760
    panel_a = (40, 95, 740, 615)
    panel_b = (800, 95, 360, 615)
    plot_left, plot_top, plot_width, plot_height = 110, 195, 635, 315
    plot_bottom = plot_top + plot_height
    selected_index = min(range(len(evidence.validation_curve)), key=lambda index: abs(evidence.validation_curve[index]["threshold"] - evidence.threshold))
    curve = _downsample_curve(evidence.validation_curve, selected_index)
    points = " ".join(
        f"{_fmt(plot_left + row['recall'] * plot_width)},{_fmt(plot_top + (1 - row['precision']) * plot_height)}"
        for row in curve
    )
    selected_x = plot_left + evidence.validation["recall"] * plot_width
    selected_y = plot_top + (1 - evidence.validation["precision"]) * plot_height
    prevalence_y = plot_top + (1 - evidence.validation_prevalence) * plot_height

    test_tp = int(evidence.test["tp"])
    test_fp = int(evidence.test["fp"])
    test_fn = int(evidence.test["fn"])
    total_alerts = test_tp + test_fp
    total_frauds = test_tp + test_fn
    bar_x, bar_width, bar_height = 830, 300, 34
    queue_tp_width = bar_width * test_tp / total_alerts
    queue_fp_width = bar_width * test_fp / total_alerts
    fraud_tp_width = bar_width * test_tp / total_frauds
    fraud_fn_width = bar_width * test_fn / total_frauds
    near_min = evidence.near_optimal["near_optimal_threshold_min"]
    near_max = evidence.near_optimal["near_optimal_threshold_max"]

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">Fraud detection operating point — {_esc(evidence.model_name)}</title>
  <desc id="description">Selected model: {_esc(evidence.model_name)}. Validation threshold: {evidence.threshold:.6f}. Untouched chronological test set: precision {_pct(evidence.test["precision"])}, recall {_pct(evidence.test["recall"])}, PR-AUC {evidence.test["average_precision"]:.4f}; {test_tp} true positives, {test_fp} false positives and {test_fn} false negatives.</desc>
  <defs>
    <pattern id="false-positive-pattern" width="8" height="8" patternUnits="userSpaceOnUse">
      <rect width="8" height="8" fill="#f0e1e8"/>
      <path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#8f5a70" stroke-width="1.2"/>
    </pattern>
    <pattern id="missed-fraud-pattern" width="8" height="8" patternUnits="userSpaceOnUse">
      <rect width="8" height="8" fill="#f1e5ce"/>
      <path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#a46d25" stroke-width="1.2"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="#fcfbf8"/>
  {_text(40, 36, "Fraud detection operating point", size=28, weight="700")}
  {_text(40, 61, "Threshold selected on validation · evaluated once on the chronological test set", size=15, fill="#59666d")}
  <rect x="{panel_a[0]}" y="{panel_a[1]}" width="{panel_a[2]}" height="{panel_a[3]}" fill="#ffffff" stroke="#d7d4ce" stroke-width="1"/>
  <rect x="{panel_b[0]}" y="{panel_b[1]}" width="{panel_b[2]}" height="{panel_b[3]}" fill="#ffffff" stroke="#d7d4ce" stroke-width="1"/>

  {_text(65, 130, "Panel A — Validation precision–recall evidence", size=21, weight="700")}
  {_text(65, 151, f"Chronological validation split · {evidence.validation_rows:,} transactions · {evidence.validation_frauds} frauds", size=13, fill="#59666d")}
  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#263238" stroke-width="1.2"/>
  <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_left + plot_width}" y2="{plot_bottom}" stroke="#263238" stroke-width="1.2"/>
  <line x1="{plot_left}" y1="{plot_top + plot_height * 0.5}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height * 0.5}" stroke="#e3e6e8" stroke-width="1"/>
  <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left + plot_width}" y2="{plot_top}" stroke="#e3e6e8" stroke-width="1"/>
  <line x1="{plot_left + plot_width * 0.5}" y1="{plot_top}" x2="{plot_left + plot_width * 0.5}" y2="{plot_bottom}" stroke="#eef0f1" stroke-width="1"/>
  <line x1="{plot_left + plot_width}" y1="{plot_top}" x2="{plot_left + plot_width}" y2="{plot_bottom}" stroke="#eef0f1" stroke-width="1"/>
  {_text(plot_left - 12, plot_top + 5, "1.0", size=12, fill="#6b7479", anchor="end")}
  {_text(plot_left - 12, plot_top + plot_height * 0.5 + 4, "0.5", size=12, fill="#6b7479", anchor="end")}
  {_text(plot_left - 12, plot_bottom + 4, "0.0", size=12, fill="#6b7479", anchor="end")}
  {_text(plot_left, plot_bottom + 22, "0.0", size=12, fill="#6b7479", anchor="middle")}
  {_text(plot_left + plot_width * 0.5, plot_bottom + 22, "0.5", size=12, fill="#6b7479", anchor="middle")}
  {_text(plot_left + plot_width, plot_bottom + 22, "1.0", size=12, fill="#6b7479", anchor="middle")}
  {_text(plot_left + plot_width / 2, plot_bottom + 43, "Recall", size=14, fill="#3f4b51", anchor="middle")}
  <text transform="translate(72 {plot_top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="14px" fill="#3f4b51">Precision</text>
  <line x1="{plot_left}" y1="{_fmt(prevalence_y)}" x2="{plot_left + plot_width}" y2="{_fmt(prevalence_y)}" stroke="#727d83" stroke-width="1.2" stroke-dasharray="6 5"/>
  <rect x="{plot_left + 10}" y="{plot_bottom - 42}" width="245" height="28" fill="#fcfbf8"/>
  {_text(plot_left + 18, plot_bottom - 23, f"Fraud prevalence baseline = {_pct(evidence.validation_prevalence, 3)}", size=12, fill="#59666d")}
  <polyline fill="none" stroke="#496a7a" stroke-width="2.4" points="{points}"/>
  <line x1="{_fmt(selected_x)}" y1="{_fmt(selected_y)}" x2="460" y2="572" stroke="#c85a32" stroke-width="1.4" stroke-dasharray="4 4"/>
  <circle cx="{_fmt(selected_x)}" cy="{_fmt(selected_y)}" r="8" fill="#ffffff" stroke="#c85a32" stroke-width="3"/>
  <circle cx="{_fmt(selected_x)}" cy="{_fmt(selected_y)}" r="3.5" fill="#c85a32"/>
  <rect x="460" y="548" width="280" height="93" fill="#fff8f4" stroke="#c85a32" stroke-width="1"/>
  {_text(476, 569, "Selected validation point", size=14, fill="#913f22", weight="700")}
  {_text(476, 588, f"threshold {evidence.threshold:.6f}", size=13, fill="#3f4b51")}
  {_text(476, 606, f"precision {_pct(evidence.validation['precision'])} · recall {_pct(evidence.validation['recall'])}", size=13, fill="#3f4b51")}
  {_text(476, 624, f"validation PR-AUC {evidence.validation['average_precision']:.4f}", size=13, fill="#3f4b51")}
  {_text(65, 585, "Higher precision means a larger share of alerts are fraud.", size=13, fill="#3f4b51")}
  {_text(65, 605, "Recall: fewer fraud transactions missed", size=13, fill="#3f4b51")}
  {_text(65, 654, f"Near-tie band: thresholds {near_min:.6f}–{near_max:.6f} stay within 5% of validation cost.", size=12, fill="#59666d")}
  {_text(65, 675, "Cost alone does not set the production point; capacity and minimum precision still need an operating decision.", size=12, fill="#59666d")}

  {_text(825, 130, "Panel B — Untouched chronological", size=19, weight="700")}
  {_text(825, 154, "test set — one operating point", size=19, weight="700")}
  {_text(825, 178, f"Frozen threshold applied once · {evidence.test_frauds} frauds / {evidence.test_rows:,} transactions", size=12, fill="#59666d")}
  {_text(880, 216, "test precision", size=11, fill="#59666d", anchor="middle")}
  {_text(880, 242, _pct(evidence.test["precision"]), size=22, fill="#263238", weight="700", anchor="middle")}
  {_text(980, 216, "test recall", size=11, fill="#59666d", anchor="middle")}
  {_text(980, 242, _pct(evidence.test["recall"]), size=22, fill="#263238", weight="700", anchor="middle")}
  {_text(1080, 216, "test PR-AUC", size=11, fill="#59666d", anchor="middle")}
  {_text(1080, 242, f"{evidence.test['average_precision']:.4f}", size=22, fill="#263238", weight="700", anchor="middle")}

  {_text(830, 288, "Review queue composition", size=15, weight="700")}
  {_text(830, 307, f"{total_alerts} total alerts", size=12, fill="#59666d")}
  <rect x="{bar_x}" y="321" width="{_fmt(queue_tp_width)}" height="{bar_height}" fill="#3f7f87"/>
  <rect x="{_fmt(bar_x + queue_tp_width)}" y="321" width="{_fmt(queue_fp_width)}" height="{bar_height}" fill="url(#false-positive-pattern)" stroke="#8f5a70" stroke-width="1"/>
  {_text(bar_x + queue_tp_width / 2, 343, f"{test_tp} fraud", size=13, fill="#ffffff", weight="700", anchor="middle")}
  {_text(bar_x + queue_tp_width + queue_fp_width / 2, 343, f"{test_fp} legit", size=11, fill="#5c3444", weight="700", anchor="middle")}
  {_text(830, 374, f"{test_tp} fraud alerts + {test_fp} legitimate reviews · {_pct(evidence.test['precision'])} precision", size=12, fill="#59666d")}

  {_text(830, 421, "Actual fraud outcomes", size=15, weight="700")}
  {_text(830, 440, f"{total_frauds} fraud transactions", size=12, fill="#59666d")}
  <rect x="{bar_x}" y="453" width="{_fmt(fraud_tp_width)}" height="{bar_height}" fill="#3f7f87"/>
  <rect x="{_fmt(bar_x + fraud_tp_width)}" y="453" width="{_fmt(fraud_fn_width)}" height="{bar_height}" fill="url(#missed-fraud-pattern)" stroke="#a46d25" stroke-width="1"/>
  {_text(bar_x + fraud_tp_width / 2, 475, f"{test_tp} caught", size=13, fill="#ffffff", weight="700", anchor="middle")}
  {_text(bar_x + fraud_tp_width + fraud_fn_width / 2, 475, f"{test_fn} missed", size=11, fill="#704b18", weight="700", anchor="middle")}
  {_text(830, 506, f"{test_tp} of {total_frauds} frauds caught · {test_fn} missed · {_pct(evidence.test['recall'])} recall", size=12, fill="#59666d")}

  {_text(830, 548, "Costs are illustrative assumptions.", size=12, fill="#3f4b51")}
  {_text(830, 568, "No real fraud-review capacity was supplied.", size=12, fill="#3f4b51")}
  {_text(830, 594, "Production threshold ownership spans fraud", size=12, fill="#59666d")}
  {_text(830, 612, "operations, risk/finance, product and data science.", size=12, fill="#59666d")}
  {_text(40, 742, "Chronological split · V1–V28 are anonymised PCA components, not customer-facing reason codes · portfolio case study, not a deployed bank fraud system.", size=12, fill="#59666d")}
</svg>
'''


def render_operating_point_svg(project_root: Path, output_path: Path | None = None) -> Path:
    project_root = Path(project_root).resolve()
    evidence = load_evidence(project_root)
    destination = Path(output_path or project_root / "reports" / "fraud_operating_point.svg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_svg(evidence))
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the deterministic fraud operating-point SVG from committed artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        destination = render_operating_point_svg(args.project_root, args.output)
    except EvidenceError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
