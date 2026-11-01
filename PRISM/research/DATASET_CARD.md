# PRISM Synthetic Benchmark

## Purpose and Provenance

All Jira tickets, PRs, authors, paths, text and labels are generated locally.
No organization data or generative LLM is used. This benchmark studies method
behavior under controlled assumptions; it is not a dataset of actual defects.

The seeded catalog has six modules and twenty task/paraphrase pairs per module.
Canonical module/task families are assigned to disjoint time windows. Broad
modules and rendering templates recur across splits, but task families and exact
task-specific text do not. Finite shared templates remain a source of artifacts.
There is one candidate ticket per task/difficulty/split. PRs reuse tasks with
independently selected Jira/PR title templates, three description renderings,
and seeded synonym substitutions. The synonym inventory remains small and
finite; this is controlled paraphrase variety, not open-ended language.

## Schema and Labels

`pull_requests.csv` contains PR/Jira IDs, four text/path fields, branch, synthetic
review timestamp, author, observable metadata, supervised labels, difficulty,
misalignment type and split/group fields. IDs and groups organize evaluation;
they are not classifier inputs. Files use JSON arrays, not Python literals.

Alignment begins with 65% correctly linked pairs. The remaining pairs are
partially related (same module, different task), unrelated (different module),
or vague. Small candidate pools can lack a same-module/different-task alternative;
such cases become vague negatives. `dataset.json` records realized counts.
Changed fields and branch describe the actual PR, not the incorrect linked issue.

Easy/medium/hard request partial-negative shares of 10%/45%/75% within the
negative branch; text-noise probabilities are 2%/10%/25%, and independent
alignment/risk label-flip probabilities are 0%/2%/5%. Noise is sampled, not forced.
Text noise is empty description, a typo transformation, or an unrelated sentence.
The typo transformation can be a no-op for some templates; richer corruption and
larger paraphrase vocabularies are research follow-ups, not validated coverage.

Risk probability is a logistic function of latent complexity, familiarity and
defect propensity plus noise. A Bernoulli draw gives the clean label; optional
label noise follows. Churn and file counts partly proxy complexity. Comments
partly proxy familiarity. Other activity features are chronological synthetic
observations, not perfect proxies or measured developer experience.

`audit.json` separately stores clean labels, latent variables, generating
probability and actual relevant issue IDs. Models never receive these columns.
The generator's public label is not overwritten by training. Audit truth is used
only for explicit evaluation and retrieval denominators.

## Splits and Leakage Controls

Chronological fractions are train 60%, selection 10%, calibration 5%, threshold
selection 5%, and test remainder (20%, with rounding). All task groups stay in a
single partition across difficulties. All candidate issues precede their queries.
Author activity considers only earlier events. No final status, post-merge data,
latent probability or label-derived shortcut is a predictor.

The validator checks identities, linked issue boundaries, retrieval availability,
group disjointness, chronology, JSON path schemas, absence of audit columns in
inputs and both classes per aggregate partition. Small per-difficulty/type subsets
can be single-class; metrics explicitly mark undefined AUC/AP. Dataset reload
checks the original SHA-256 fingerprint. Input editing requires a new benchmark,
not silently reusing a prior split manifest.

## Evaluation and Limits

Risk and alignment have separate feature contracts and outcomes. Risk does not
consume alignment predictions. Vocabulary/IDF and model fitting use training only;
development partitions control selection, calibration and thresholds.

Ranking candidate pools are split/difficulty-local and timestamp-filtered.
Queries without a relevant ticket are excluded from MRR/Recall@k, with counts
reported. Ranking evaluates the bi-encoder, not the final classifier. Clean-label
scores and noisy-label scores are distinct. Bootstrap intervals resample task
groups; multi-seed summaries measure generation/training variability.

The author controls generator assumptions and models. Shared templates,
finite tasks, simplistic module/path correspondence, non-empirical risk factors,
and limited fine-tuning diversity can inflate or distort scores. Very high
synthetic alignment scores should prompt harder held-out data, not claims of
real-world accuracy. Real-data validation is future work. SHAP describes modeled
associations in the forest, not causation or calibrated-model explanations.