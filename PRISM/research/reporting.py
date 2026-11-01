"""Escaped static reviewer reports and headless visual assets."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment
from sklearn.calibration import calibration_curve


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PRISM research results</title><style>
:root{--ink:#202a28;--muted:#52615b;--line:#d8dfdb;--accent:#176348;--alert:#a23d36}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font:15px Aptos,Calibri,sans-serif;
background:repeating-linear-gradient(0deg,#f7faf8,#f7faf8 23px,#f4f7f5 24px);letter-spacing:0}
main{max-width:1120px;margin:auto;padding:28px 24px}header{border-bottom:3px solid var(--accent);padding-bottom:18px}
h1{font:32px Georgia,serif;margin:0 0 8px}h2{font-size:20px;margin:28px 0 12px}
p{line-height:1.5;max-width:85ch}section{padding-bottom:20px;border-bottom:1px solid var(--line)}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:650px;background:#fff}
th,td{padding:10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}
th{color:var(--accent);font-weight:600;white-space:nowrap}td:first-child{min-width:130px}
figure{margin:20px 0}img{display:block;width:100%;max-width:800px;height:auto}
.notice{color:var(--alert)}code{overflow-wrap:anywhere}.number{font-variant-numeric:tabular-nums;white-space:nowrap}
@media(max-width:600px){main{padding:20px 12px}th,td{padding:8px}h1{font-size:28px}}
</style></head><body><main><header><h1>PRISM</h1><p>Semantic alignment and change-risk research results</p>
<p>{{ mode }} | Seed {{ seed }} | Held-out test records: {{ metrics.overall.risk.records }}</p></header>
<section><h2>Held-Out Evaluation</h2><div class="table-wrap"><table><thead><tr><th>Task</th><th>AP</th><th>F1</th><th>Brier</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
{% for name, result in tasks %}<tr><td>{{ name }}</td>{% for key in ['average_precision','f1','brier','precision','recall'] %}<td class="number">{{ '%.3f'|format(result[key]) if result[key] is not none else 'N/A' }}</td>{% endfor %}</tr>{% endfor %}
</tbody></table></div><figure><img src="calibration.png" alt="Held-out reliability curves for alignment and risk"></figure></section>
<section><h2>Difficulty Breakdown</h2><div class="table-wrap"><table><tr><th>Difficulty</th><th>Records</th><th>Misalignment AP</th><th>Risk AP</th></tr>
{% for name, result in metrics.difficulty.items() %}<tr><td>{{ name }}</td><td>{{ result.risk.records }}</td><td>{{ result.alignment_misaligned_positive.average_precision }}</td><td>{{ result.risk.average_precision }}</td></tr>{% endfor %}</table></div></section>
<section><h2>Classification Error Analysis</h2><div class="table-wrap"><table><tr><th>Task</th><th>False positives</th><th>False negatives</th><th>False negatives by difficulty</th><th>False negatives by type</th></tr>
{% for name, item in metrics.error_analysis.items() %}<tr><td>{{ name }}</td><td>{{ item.false_positive_count }}</td><td>{{ item.false_negative_count }}</td><td>{{ item.false_negatives_by_difficulty }}</td><td>{{ item.false_negatives_by_misalignment_type }}</td></tr>{% endfor %}</table></div>
{% for name, item in metrics.error_analysis.items() %}<h3>{{ name }} false-positive examples</h3><div class="table-wrap"><table><tr><th>PR</th><th>Title</th><th>Difficulty</th><th>Type</th><th>Predicted probability</th></tr>{% for row in item.false_positive_examples %}<tr><td>{{ row.pr_id }}</td><td>{{ row.title }}</td><td>{{ row.difficulty }}</td><td>{{ row.misalignment_type }}</td><td>{{ '%.3f'|format(row.probability) }}</td></tr>{% endfor %}</table></div>
<h3>{{ name }} false-negative examples</h3><div class="table-wrap"><table><tr><th>PR</th><th>Title</th><th>Difficulty</th><th>Type</th><th>Predicted probability</th></tr>{% for row in item.false_negative_examples %}<tr><td>{{ row.pr_id }}</td><td>{{ row.title }}</td><td>{{ row.difficulty }}</td><td>{{ row.misalignment_type }}</td><td>{{ '%.3f'|format(row.probability) }}</td></tr>{% endfor %}</table></div>{% endfor %}</section>
<section><h2>Review Queue</h2><p>Selected alignment model: {{ alignment_name }}. Selected risk model: {{ risk_name }}.</p>
<p>Contributions below explain the separate, uncalibrated Random Forest baseline, not the selected calibrated model. They are not causal explanations.</p>
<div class="table-wrap"><table><thead><tr><th>PR</th><th>Title</th><th>Aligned Probability</th><th>Risk Probability</th><th>Decision</th><th>Forest Contributions</th></tr></thead><tbody>
{% for row in queue %}<tr><td>{{ row.pr_id }}</td><td>{{ row.title }}</td><td class="number">{{ '%.3f'|format(row.alignment_probability) }}</td><td class="number">{{ '%.3f'|format(row.risk_probability) }}</td><td>{{ row.decision }}</td><td>{{ row.explanation }}</td></tr>{% endfor %}
</tbody></table></div></section><section><h2>Research Limits</h2>
<section><h2>All Held-Out PRs</h2><details><summary>{{ all_test|length }} test PRs with per-record risk contributions</summary><div class="table-wrap"><table><thead><tr><th>PR</th><th>Title</th><th>Aligned probability</th><th>Risk probability</th><th>Decision</th><th>Top risk contributions</th></tr></thead><tbody>
{% for row in all_test %}<tr><td>{{ row.pr_id }}</td><td>{{ row.title }}</td><td class="number">{{ '%.3f'|format(row.alignment_probability) }}</td><td class="number">{{ '%.3f'|format(row.risk_probability) }}</td><td>{{ row.decision }}</td><td>{{ row.explanation }}</td></tr>{% endfor %}
</tbody></table></div></details></section><section><h2>Research Limits</h2>
<p class="notice">Synthetic benchmark only. Risk labels come from assumed latent factors, not observed software defects. Results do not demonstrate production performance.</p>
<p>AP is the average-precision summary of the precision-recall curve. Calibration and thresholds used development data only. Single-class subset AUCs are undefined. Exact reruns can differ slightly across GPU/library versions.</p>
<p>Full metrics, split manifests, model artifacts, and per-PR results accompany this report.</p></section></main></body></html>"""


def write_report(output, rows, alignment_probability, risk_probability, metrics,
                 alignment_model, risk_model, explanations, mode, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows):
        if row["split"] != "test":
            continue
        explanation = explanations.get(row["pr_id"], {})
        top = sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        alerts = []
        if 1 - alignment_probability[index] >= alignment_model.threshold:
            alerts.append("Check alignment")
        if risk_probability[index] >= risk_model.threshold:
            alerts.append("Prioritize review")
        results.append({"pr_id": row["pr_id"], "jira_id": row["jira_id"], "title": row["title"],
                        "split": row["split"], "difficulty": row["difficulty"],
                        "alignment_probability": float(alignment_probability[index]),
                        "risk_probability": float(risk_probability[index]),
                        "decision": "; ".join(alerts) or "Routine review",
                        "explanation": "; ".join(f"{name}: {value:+.3f}" for name, value in top)})
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    figure, axes = plt.subplots(figsize=(8, 4.5))
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    for label, probabilities, target, color in (
        ("Risk", risk_probability, "is_high_risk", "#176348"),
        ("Aligned", alignment_probability, "is_aligned", "#b1582c"),
    ):
        observed, predicted = calibration_curve([rows[index][target] for index in test_indices],
                                                [probabilities[index] for index in test_indices], n_bins=5)
        axes.plot(predicted, observed, "o-", label=label, color=color)
    axes.plot([0, 1], [0, 1], "--", color="#87948d", label="Ideal")
    axes.set(xlabel="Predicted probability", ylabel="Observed fraction", xlim=(0, 1), ylim=(0, 1))
    axes.legend()
    figure.tight_layout()
    figure.savefig(output / "calibration.png", dpi=150)
    plt.close(figure)
    queue = sorted(results,
                   key=lambda row: max(row["risk_probability"], 1 - row["alignment_probability"]), reverse=True)[:30]
    rendered = Environment(autoescape=True).from_string(TEMPLATE).render(
        metrics=metrics, mode=mode, seed=seed, queue=queue, all_test=results,
        alignment_name=alignment_model.name, risk_name=risk_model.name,
        tasks=[("Misalignment", metrics["overall"]["alignment_misaligned_positive"]),
               ("High risk", metrics["overall"]["risk"])],
    )
    (output / "report.html").write_text(rendered, encoding="utf-8")
    return results