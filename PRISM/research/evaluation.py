"""Held-out classification, retrieval, and grouped uncertainty summaries."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not len(labels):
        return {"records": 0}
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Metric inputs must be matching finite vectors")
    predictions = probabilities >= threshold
    both_classes = set(labels) == {0, 1}
    return {
        "records": len(labels), "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "warning": None if both_classes else "Single-class subset: AUC/AP undefined",
    }


def retrieval_metrics(ranks):
    eligible = [rank for rank in ranks if rank is not None]
    result = {"eligible_queries": len(eligible), "excluded_no_relevant_ticket": len(ranks) - len(eligible)}
    result["mrr"] = float(np.mean([1 / rank if rank else 0 for rank in eligible])) if eligible else None
    for cutoff in (1, 3, 5):
        result[f"recall_at_{cutoff}"] = float(np.mean([0 < rank <= cutoff for rank in eligible])) if eligible else None
    return result


def rank_queries(encoder, rows, tickets, audit):
    from .alignment import issue_text, pr_text
    lookup = {item["pr_id"]: item for item in audit}
    ranks = []
    for row in rows:
        relevant = lookup[row["pr_id"]]["relevant_jira_ids"]
        if not relevant:
            ranks.append(None)
            continue
        candidates = [ticket for ticket in tickets if ticket["split"] == row["split"]
                      and ticket["difficulty"] == row["difficulty"]
                      and ticket["created_date"] <= row["created_date"]]
        scores = encoder.encode([issue_text(ticket) for ticket in candidates]) @ encoder.encode([pr_text(row)])[0]
        ordered = np.argsort(-scores, kind="stable")
        matches = [position + 1 for position, index in enumerate(ordered) if candidates[index]["id"] in relevant]
        ranks.append(min(matches) if matches else 0)
    return ranks


def grouped_ap_interval(labels, probabilities, groups, seed=42, repetitions=100):
    rng = np.random.default_rng(seed)
    labels, probabilities, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(groups)
    unique = np.unique(groups)
    scores = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if set(labels[indices]) == {0, 1}:
            scores.append(average_precision_score(labels[indices], probabilities[indices]))
    return {"lower": float(np.percentile(scores, 2.5)), "upper": float(np.percentile(scores, 97.5)),
            "valid_resamples": len(scores)} if scores else None


def error_analysis(rows, labels, probabilities, threshold, positive_name):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= threshold
    false_positive = np.flatnonzero((labels == 0) & predictions)
    false_negative = np.flatnonzero((labels == 1) & ~predictions)

    def examples(indices):
        return [{"pr_id": rows[index]["pr_id"], "title": rows[index]["title"],
                 "difficulty": rows[index]["difficulty"],
                 "misalignment_type": rows[index]["misalignment_type"],
                 "probability": float(probabilities[index]),
                 "label": int(labels[index]), "predicted": int(predictions[index])}
                for index in indices[:10]]

    def counts(indices, key):
        result = {}
        for index in indices:
            value = rows[index][key]
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "positive_class": positive_name,
        "threshold": float(threshold),
        "false_positive_count": int(len(false_positive)),
        "false_negative_count": int(len(false_negative)),
        "false_positives_by_difficulty": counts(false_positive, "difficulty"),
        "false_negatives_by_difficulty": counts(false_negative, "difficulty"),
        "false_positives_by_misalignment_type": counts(false_positive, "misalignment_type"),
        "false_negatives_by_misalignment_type": counts(false_negative, "misalignment_type"),
        "false_positive_examples": examples(false_positive),
        "false_negative_examples": examples(false_negative),
    }


def evaluate(rows, audit, alignment_probability, risk_probability, alignment_threshold,
             risk_threshold, ranks=None, seed=42):
    test = np.array([row["split"] == "test" for row in rows])
    alignment_probability, risk_probability = np.asarray(alignment_probability), np.asarray(risk_probability)
    aligned = np.array([row["is_aligned"] for row in rows])
    risk = np.array([row["is_high_risk"] for row in rows])
    lookup = {item["pr_id"]: item for item in audit}

    def summary(mask):
        result = {
            "alignment_misaligned_positive": binary_metrics(1 - aligned[mask], 1 - alignment_probability[mask], alignment_threshold),
            "alignment_aligned_positive": binary_metrics(aligned[mask], alignment_probability[mask], 1 - alignment_threshold),
            "risk": binary_metrics(risk[mask], risk_probability[mask], risk_threshold),
        }
        if ranks is not None:
            result["bi_encoder_retrieval"] = retrieval_metrics([rank for rank, keep in zip(ranks, mask) if keep])
        return result

    test_rows = [row for row, keep in zip(rows, test) if keep]
    result = {"evaluation_split": "test", "pr_auc_summary": "average precision (AP)",
              "overall": summary(test), "difficulty": {}, "misalignment_type": {}}
    result["error_analysis"] = {
        "misalignment": error_analysis(test_rows, 1 - aligned[test], 1 - alignment_probability[test],
                                        alignment_threshold, "misaligned"),
        "risk": error_analysis(test_rows, risk[test], risk_probability[test], risk_threshold, "high risk"),
    }
    for key in ("difficulty", "misalignment_type"):
        for value in sorted({row[key] for row in rows}):
            mask = test & np.array([row[key] == value for row in rows])
            result[key][value] = summary(mask)
    clean_aligned = np.array([lookup[row["pr_id"]]["clean_is_aligned"] for row in rows])
    clean_risk = np.array([lookup[row["pr_id"]]["clean_is_high_risk"] for row in rows])
    result["clean_label_audit"] = {
        "alignment": binary_metrics(1 - clean_aligned[test], 1 - alignment_probability[test], alignment_threshold),
        "risk": binary_metrics(clean_risk[test], risk_probability[test], risk_threshold),
    }
    groups = [row["group_id"] for row in rows if row["split"] == "test"]
    result["group_bootstrap_95_percent"] = {
        "risk_ap": grouped_ap_interval(risk[test], risk_probability[test], groups, seed),
        "misalignment_ap": grouped_ap_interval(1 - aligned[test], 1 - alignment_probability[test], groups, seed),
    }
    return result