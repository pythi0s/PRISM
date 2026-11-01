import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from PRISM.research.data import generate_benchmark
from PRISM.research.evaluation import binary_metrics, evaluate, retrieval_metrics
from PRISM.research.modeling import CalibratedModel
from PRISM.research.reporting import write_report


class EvaluationTests(unittest.TestCase):
    def test_single_class_and_retrieval_denominator(self):
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2])["roc_auc"])
        result = retrieval_metrics([1, 2, None, 0])
        self.assertEqual(result["eligible_queries"], 3)
        self.assertAlmostEqual(result["mrr"], 0.5)
        self.assertEqual(result["excluded_no_relevant_ticket"], 1)

    def test_report_escape_and_real_asset(self):
        rows, _, audit = generate_benchmark()
        for row in rows:
            row["title"] = "<script>alert(1)</script>"
        probabilities = np.full(len(rows), 0.5)
        metrics = evaluate(rows, audit, probabilities, probabilities, 0.5, 0.5)
        model = CalibratedModel(None, None, 0.5, [], "test", {})
        explanations = {row["pr_id"]: {"observable_feature": 0.125}
                        for row in rows if row["split"] == "test"}
        with tempfile.TemporaryDirectory() as folder:
            write_report(folder, rows, probabilities, probabilities, metrics,
                         model, model, explanations, "Explicit test fixture", 42)
            content = (Path(folder) / "report.html").read_text()
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertGreater((Path(folder) / "calibration.png").stat().st_size, 1000)
            self.assertEqual(metrics["overall"]["risk"]["records"], 60)
            self.assertIn("Classification Error Analysis", content)
            self.assertIn("All Held-Out PRs", content)
            with (Path(folder) / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 60)
            self.assertTrue(all(row["explanation"] for row in predictions))
            for task in ("risk", "misalignment"):
                analysis = metrics["error_analysis"][task]
                self.assertEqual(analysis["false_positive_count"],
                                 sum(analysis["false_positives_by_difficulty"].values()))
                self.assertEqual(analysis["false_negative_count"],
                                 sum(analysis["false_negatives_by_difficulty"].values()))


if __name__ == "__main__":
    unittest.main()