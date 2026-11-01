"""Shared held-out model selection, calibration, and decision thresholds."""

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def log_odds(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@dataclass
class CalibratedModel:
    estimator: object
    calibrator: object
    threshold: float
    feature_names: list
    name: str
    comparisons: dict

    def predict_probability(self, features):
        if features.shape[1] != len(self.feature_names):
            raise ValueError("Inference feature schema differs from training")
        raw = self.estimator.predict_proba(features)[:, 1]
        return self.calibrator.predict_proba(log_odds(raw))[:, 1]


@threadpool_limits.wrap(limits=1)
def train_classifiers(features, rows, label, feature_names, seed=42, invert=False,
                      include_forest=True, cross_validate=False):
    if not np.isfinite(features).all():
        raise ValueError("Model features must be finite")
    labels = np.array([row[label] for row in rows])
    if invert:
        labels = 1 - labels
    masks = {split: np.array([row["split"] == split for row in rows])
             for split in ("train", "selection", "calibration", "threshold")}
    for split, mask in masks.items():
        if set(labels[mask]) != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=seed),
    }
    if include_forest:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=50, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", n_jobs=1, random_state=seed,
        )
    comparisons, fitted = {}, {}
    for name, estimator in models.items():
        pipeline = make_pipeline(SimpleImputer(), StandardScaler(), estimator)
        results = {}
        if cross_validate:
            groups = np.array([row["group_id"] for row in rows])[masks["train"]]
            folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            scores = cross_val_score(pipeline, features[masks["train"]], labels[masks["train"]],
                                     groups=groups, cv=folds, scoring="average_precision", error_score="raise",
                                     n_jobs=1, pre_dispatch=1)
            results["training_cv_ap_mean"] = float(scores.mean())
            results["training_cv_ap_std"] = float(scores.std())
        pipeline.fit(features[masks["train"]], labels[masks["train"]])
        probabilities = pipeline.predict_proba(features[masks["selection"]])[:, 1]
        results["selection_ap"] = float(average_precision_score(labels[masks["selection"]], probabilities))
        comparisons[name] = results
        fitted[name] = pipeline
    best = max(comparisons, key=lambda name: comparisons[name]["selection_ap"])
    estimator = fitted[best]
    calibration_raw = estimator.predict_proba(features[masks["calibration"]])[:, 1]
    calibrator = LogisticRegression(C=1.0, random_state=seed).fit(
        log_odds(calibration_raw), labels[masks["calibration"]],
    )
    probabilities = calibrator.predict_proba(log_odds(
        estimator.predict_proba(features[masks["threshold"]])[:, 1],
    ))[:, 1]
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    threshold = max(thresholds, key=lambda value: (
        f1_score(labels[masks["threshold"]], probabilities >= value, zero_division=0),
        -abs(value - 0.5),
    ))
    bundle = CalibratedModel(estimator, calibrator, float(threshold), list(feature_names), best, comparisons)
    return bundle, fitted