"""Risk prediction from an explicit observable-only feature contract."""

import numpy as np

from .data import RISK_FEATURES, parse_files
from .modeling import train_classifiers


FEATURE_NAMES = [*RISK_FEATURES, "churn_ratio"]


def risk_features(rows):
    values = []
    for row in rows:
        parse_files(row["files_changed"])
        features = [float(row[name]) for name in RISK_FEATURES]
        if any(value < 0 for value in features):
            raise ValueError("Risk metadata cannot be negative")
        features.append(row["lines_added"] / (row["lines_deleted"] + 1))
        values.append(features)
    return np.array(values, dtype=float)


def train_risk(rows, seed=42):
    features = risk_features(rows)
    bundle, baselines = train_classifiers(features, rows, "is_high_risk", FEATURE_NAMES,
                                        seed=seed, cross_validate=True)
    return features, bundle, baselines


def explain_forest(forest, training_features, explain_features, feature_names, batch_size=64):
    import shap
    if not len(training_features) or batch_size < 1:
        raise ValueError("Explanations require a trained forest")
    preprocessing = forest[:-1]
    explainer = shap.TreeExplainer(
        forest[-1], feature_perturbation="tree_path_dependent", model_output="raw",
    )
    transformed = preprocessing.transform(explain_features)
    all_values = []
    for start in range(0, len(transformed), batch_size):
        stop = start + batch_size
        explanation = explainer(transformed[start:stop], check_additivity=True)
        values = explanation.values[:, :, 1]
        reconstructed = explanation.base_values[:, 1] + values.sum(axis=1)
        expected = forest.predict_proba(explain_features[start:stop])[:, 1]
        if not np.allclose(reconstructed, expected, atol=1e-5):
            raise ValueError("SHAP probability contributions failed additivity verification")
        all_values.extend(values)
    return [{feature_names[index]: float(contribution) for index, contribution in enumerate(row)}
            for row in all_values]