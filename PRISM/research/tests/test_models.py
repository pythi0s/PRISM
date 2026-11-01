import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from threadpoolctl import threadpool_info

from PRISM.research.data import generate_benchmark
from PRISM.research.risk import FEATURE_NAMES, explain_forest, risk_features, train_risk


class RiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_benchmark(100, 42)[0]
        cls.features, cls.model, cls.baselines = train_risk(cls.rows)

    def test_whitelist_and_finite_probability(self):
        self.assertEqual(self.features.shape, (300, len(FEATURE_NAMES)))
        probabilities = self.model.predict_probability(self.features)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        changed = [dict(row, latent_complexity=100, risk_probability=1) for row in self.rows]
        np.testing.assert_equal(self.features, risk_features(changed))

    def test_test_labels_cannot_change_training(self):
        changed = [dict(row) for row in self.rows]
        for row in changed:
            if row["split"] == "test":
                row["is_high_risk"] = 1 - row["is_high_risk"]
        _, model, _ = train_risk(changed)
        np.testing.assert_allclose(self.model.predict_probability(self.features),
                                   model.predict_probability(self.features))

    def test_persistence_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(self.model, path)
            reloaded = joblib.load(path)
            np.testing.assert_allclose(self.model.predict_probability(self.features),
                                       reloaded.predict_probability(self.features))
        with self.assertRaises(ValueError):
            self.model.predict_probability(self.features[:, :-1])

    def test_forest_explanations(self):
        train = np.array([row["split"] == "train" for row in self.rows])
        explanations = explain_forest(self.baselines["random_forest"], self.features[train],
                                      self.features[:2], FEATURE_NAMES)
        self.assertEqual(len(explanations), 2)
        self.assertEqual(set(explanations[0]), set(FEATURE_NAMES))

    def test_model_fitting_is_single_threaded(self):
        observed = []
        original = GradientBoostingClassifier.fit

        def checked_fit(estimator, *args, **kwargs):
            observed.extend(pool["num_threads"] for pool in threadpool_info())
            return original(estimator, *args, **kwargs)

        with patch.object(GradientBoostingClassifier, "fit", checked_fit):
            train_risk(self.rows)
        self.assertTrue(observed)
        self.assertLessEqual(max(observed), 1)
        self.assertEqual(self.baselines["random_forest"][-1].n_jobs, 1)


if __name__ == "__main__":
    unittest.main()