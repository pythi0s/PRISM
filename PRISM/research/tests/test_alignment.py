import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from PRISM.research.alignment import AlignmentFeatures, mine_negative_indices, train_alignment
from PRISM.research.data import generate_benchmark, write_benchmark


class TestEncoder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] if text else [0.0, 0.0] for text in texts])


class TestCrossEncoder:
    def predict(self, pairs):
        return np.zeros(len(pairs))


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.tickets, _ = generate_benchmark(100, 42)

    def test_feature_dimensions_and_empty_fields(self):
        extractor = AlignmentFeatures().fit(self.rows, self.tickets)
        values, names = extractor.transform(self.rows, self.tickets, TestEncoder(), TestCrossEncoder())
        self.assertEqual(values.shape, (300, 11))
        self.assertEqual(len(names), 11)
        changed = dict(self.rows[0], description="")
        empty, _ = extractor.transform([changed], self.tickets, TestEncoder())
        self.assertEqual(empty[0, names.index("semantic_description")], 0)
        self.assertTrue(np.isfinite(empty).all())

    def test_test_text_does_not_fit_vocabulary(self):
        rows = [dict(row) for row in self.rows]
        for row in rows:
            if row["split"] == "test":
                row["description"] = "heldoutonlytoken"
        extractor = AlignmentFeatures().fit(rows, self.tickets)
        self.assertNotIn("heldoutonlytoken", extractor.vectorizer.vocabulary_)
        self.assertEqual(extractor.bm25.avgdl,
                         AlignmentFeatures().fit(self.rows, self.tickets).bm25.avgdl)

    def test_classifier_and_negative_filter(self):
        extractor = AlignmentFeatures().fit(self.rows, self.tickets)
        values, names = extractor.transform(self.rows, self.tickets)
        model, _ = train_alignment(values, self.rows, names)
        self.assertTrue(np.isfinite(model.predict_probability(values)).all())
        self.assertEqual(mine_negative_indices([0.9, 0.8, 0.7], ["a", "b", "c"], {"a", "b"}), [2])

    def test_checkpoint_reuse_does_not_retrain(self):
        from PRISM.research.cli import main

        class FakeBiEncoder(TestEncoder):
            def __init__(self, *args, **kwargs):
                self.gpu_preflight = None
                self.model = {0: SimpleNamespace(auto_model=SimpleNamespace(
                    config=SimpleNamespace(_commit_hash="test-revision")))}

            def unload(self):
                return None

        class FakeCrossEncoder(TestCrossEncoder):
            model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision"))

            def predict(self, pairs, **kwargs):
                return super().predict(pairs)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            rows, tickets, audit = generate_benchmark()
            write_benchmark(output, rows, tickets, audit, {"seed": 42})
            (output / "bge_finetuned").mkdir()
            (output / "finetuning.json").write_text("[]")
            with patch("PRISM.research.alignment.BiEncoder", FakeBiEncoder), \
                    patch("PRISM.research.alignment.load_cross_encoder", return_value=FakeCrossEncoder()), \
                    patch("PRISM.research.alignment.finetune") as training:
                main(["features", "--output", str(output), "--with-encoders",
                      "--reuse-finetuned", "--allow-intensive", "--device", "cpu", "--batch-size", "1"])
                training.assert_not_called()
            metadata = json.loads((output / "features.json").read_text())
            self.assertTrue(metadata["runtime"]["reused_finetuned_checkpoint"])
            self.assertIn("finetuned_hybrid", metadata["feature_names"])


if __name__ == "__main__":
    unittest.main()