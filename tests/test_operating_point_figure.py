from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.operating_point_figure import EvidenceError, load_evidence, render_operating_point_svg


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path) -> None:
    (root / "artifacts").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    artifact = {
        "model_name": "unweighted_logistic",
        "threshold": 0.5,
        "threshold_selection": {
            "near_optimal_threshold_min": 0.5,
            "near_optimal_threshold_max": 0.6,
            "near_optimal_precision_min": 0.6,
            "near_optimal_precision_max": 0.75,
            "near_optimal_recall_min": 0.75,
            "near_optimal_recall_max": 1.0,
        },
        "metrics": {
            "validation": {"tp": 3, "fp": 1, "tn": 5, "fn": 1, "precision": 0.75, "recall": 0.75, "average_precision": 0.8, "threshold": 0.5},
            "test": {"tp": 2, "fp": 1, "tn": 5, "fn": 2, "precision": 2 / 3, "recall": 0.5, "average_precision": 0.7, "threshold": 0.5},
        },
    }
    (root / "artifacts" / "fraud_model.json").write_text(json.dumps(artifact), encoding="utf-8")
    metric_fields = [
        "model", "split", "tp", "fp", "tn", "fn", "precision", "recall", "accuracy", "flagged_rate",
        "false_positive_rate", "average_precision", "roc_auc", "threshold", "alerts",
    ]
    rows = [
        ["unweighted_logistic", "validation", 3, 1, 5, 1, 0.75, 0.75, 0.8, 0.4, 1 / 6, 0.8, 0.9, 0.5, 4],
        ["unweighted_logistic", "test", 2, 1, 5, 2, 2 / 3, 0.5, 0.7, 0.3, 1 / 6, 0.7, 0.85, 0.5, 3],
    ]
    with (root / "reports" / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(metric_fields)
        writer.writerows(rows)
    with (root / "reports" / "class_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "rows", "legitimate", "fraud", "fraud_rate"])
        writer.writerows([["validation", 10, 6, 4, 0.4], ["test", 10, 6, 4, 0.4]])
    with (root / "reports" / "validation_precision_recall_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "precision", "recall", "tp", "fp"])
        writer.writerows([[0.9, 1.0, 0.25, 1, 0], [0.5, 0.75, 0.75, 3, 1], [0.2, 0.4, 1.0, 4, 6]])
    (root / "reports" / "model_report.md").write_text(
        "Champion: `unweighted_logistic`\nchronological split. The test set is used only after the threshold is chosen. "
        "Costs are illustrative. The fields are not customer-facing reason codes.\n",
        encoding="utf-8",
    )
    (root / "RUN_STATUS.md").write_text("near-tie; test precision is reported\n", encoding="utf-8")
    (root / "MONITORING.md").write_text("illustrative assumptions; ownership is explicit\n", encoding="utf-8")


class OperatingPointFigureTests(unittest.TestCase):
    def test_evidence_extracts_split_support_prevalence_and_selected_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            evidence = load_evidence(root)
            self.assertEqual(evidence.model_name, "unweighted_logistic")
            self.assertEqual(evidence.threshold, 0.5)
            self.assertEqual(evidence.validation_rows, 10)
            self.assertEqual(evidence.validation_frauds, 4)
            self.assertAlmostEqual(evidence.validation_prevalence, 0.4)
            self.assertEqual(evidence.test["tp"], 2)
            self.assertEqual(evidence.test["fn"], 2)

    def test_render_is_byte_identical_and_keeps_validation_test_evidence_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            first = render_operating_point_svg(root, root / "first.svg")
            second = render_operating_point_svg(root, root / "second.svg")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            svg = first.read_text(encoding="utf-8")
            self.assertIn("threshold 0.500000", svg)
            self.assertIn("precision 75.0% · recall 75.0%", svg)
            self.assertIn("test precision", svg)
            self.assertIn("66.7%", svg)
            self.assertIn("Selected model: unweighted_logistic.", svg)
            self.assertIn("Higher precision means a larger share of alerts are fraud.", svg)
            self.assertNotIn("Precision: fewer legitimate transactions entering review", svg)
            self.assertNotIn("test set is used only after", svg)
            self.assertNotIn("created_at", svg)

    def test_committed_figure_matches_current_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = render_operating_point_svg(ROOT, Path(directory) / "fraud_operating_point.svg")
            committed = ROOT / "reports" / "fraud_operating_point.svg"
            self.assertEqual(generated.read_bytes(), committed.read_bytes())

    def test_malformed_curve_probability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            curve_path = root / "reports" / "validation_precision_recall_curve.csv"
            content = curve_path.read_text(encoding="utf-8").replace("0.75,0.75", "1.25,0.75")
            curve_path.write_text(content, encoding="utf-8")
            with self.assertRaises(EvidenceError):
                load_evidence(root)

    def test_confusion_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            metrics_path = root / "reports" / "metrics_summary.csv"
            content = metrics_path.read_text(encoding="utf-8").replace("unweighted_logistic,test,2,1,5,2", "unweighted_logistic,test,2,2,4,2")
            metrics_path.write_text(content, encoding="utf-8")
            with self.assertRaises(EvidenceError):
                load_evidence(root)

    def test_readme_points_to_committed_figure_and_keeps_operational_boundaries(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Verified operating point", readme)
        self.assertIn("[![Validation precision-recall curve", readme)
        self.assertIn("](reports/model_report.md)", readme)
        self.assertTrue((ROOT / "reports" / "fraud_operating_point.svg").is_file())
        self.assertIn("reports/model_report.md", readme)
        self.assertIn("illustrative", readme.lower())
        self.assertIn("No real fraud-review capacity", readme)
        self.assertIn("V1–V28", readme)


if __name__ == "__main__":
    unittest.main()
