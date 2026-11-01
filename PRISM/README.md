# PRISM
Pull Request Intelligence for Semantic Mapping: a controlled synthetic research
pipeline for Jira/PR alignment and change-risk estimation. The original scripts
and CSVs are preserved as a legacy prototype. Use `research/` for the new pipeline.

## Environment

Run from the repository root (the directory containing the inner `PRISM/` folder).
The verified environment is WSL/Linux, Python 3.10. The default low-resource
pipeline needs no GPU, PyTorch, transformer model downloads or API credentials.
Installation requires internet access. Do not resume the interrupted intensive
pilot automatically; its partial artifacts are preserved.

## One-Command CPU Setup

For a fresh Linux/WSL checkout, install Python 3.10 or newer (below 3.14), Bash,
and Git LFS. Ensure the matching Python `venv`/`ensurepip` support is installed.
Internet access is needed for Python packages and the Git LFS model archive.
Allow roughly 2 GB for the archive plus the extracted models, and additional disk
space for the virtual environment. Then run from the repository root:

```bash
bash PRISM/research/setup_cpu.sh
source .venv/bin/activate
```

The script creates or reuses `.venv`, installs the CPU encoder dependencies if
needed, verifies the archive checksum, and restores the three trained seeds. It
never starts training. It preserves an existing working environment; if that
environment contains CUDA-enabled Torch, commands still hide CUDA and select
CPU. For a fresh setup, the encoder manifest installs CPU-only Torch.

The trained models are distributed as a Git LFS archive:
`PRISM/research/artifacts/full_bge_cpu.tar.gz`. Verify and restore manually with:

```bash
sha256sum -c PRISM/research/artifacts/full_bge_cpu.tar.gz.sha256
tar -xzf PRISM/research/artifacts/full_bge_cpu.tar.gz -C PRISM/research/artifacts
```

The `.venv` and extracted/generated artifacts are intentionally excluded from
ordinary Git objects. The archive itself is tracked with Git LFS to avoid placing
a 1.1 GB model archive in normal Git history. A Git host account with sufficient
LFS storage/bandwidth is required to clone the archive. Back up the archive
outside the checkout for additional protection.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r PRISM/research/requirements.txt
.venv/bin/python -m pip check
```

On native Windows, use `py -3.10 -m venv .venv` and replace `.venv/bin/python`
with `.venv\Scripts\python.exe` in commands. Windows execution is not yet tested.

Optional small CPU encoders require an additional install; the manifest installs
the CPU-only PyTorch wheel, with no CUDA runtime:

```bash
.venv/bin/python -m pip install -r PRISM/research/requirements-encoders.txt
```

CPU is the default and has no GPU dependency. CUDA is optional and available
only as an explicit, VRAM-guarded opt-in. The standard CPU encoder manifest is
separate from the CUDA-specific installation below. Always use the environment
interpreter, not global `pip`, and select it in VS Code.

`research/requirements.lock` pins the verified Linux core dependency closure,
without PyTorch, transformer or CUDA packages. Use the direct `requirements.txt`
for another OS; native Windows has not been exercised here. Do not copy `.venv`
between laptops; create a fresh environment using the documented commands.

## Optional CUDA Training

GPU use is never automatic. The CUDA path requires `--allow-intensive`,
`--device cuda`, and `--batch-size 1`. It refuses to load a model unless the
GPU has at least 6,000 MiB total and at least 4 GiB free for inference or 5,000 MiB free
for fine-tuning. The PyTorch allocator is capped at 70% of VRAM. CUDA fine-tuning
uses FP16 autocast, one-sample batches, and 128-token maximum inputs; model stages
release cached VRAM before loading the next encoder. An out-of-memory failure
stops the run without an automatic retry. These limits reduce risk but cannot
guarantee protection from driver, thermal, operating-system, or unrelated GPU
failures. Keep the laptop on AC power and close GPU-heavy applications.

Install the CUDA build only on a CUDA-capable system with a compatible NVIDIA
driver. This wheel and its libraries require substantially more disk space than
the CPU-only environment:

```bash
.venv/bin/python -m pip install -r PRISM/research/requirements-encoders-cuda-cu128.txt
```

The verified RTX A3000 has 6 GiB total VRAM; its last idle reading had about
5.85 GiB free and meets the current fine-tuning preflight. This is a capacity
check only; no training was started by the preflight. If it reports insufficient
free memory, stop GPU work and use CPU rather than lowering the guard.

Conservative one-seed pilot command (one epoch, batch size 1):

```bash
.venv/bin/python -m PRISM.research.cli run-all \
	--profile pilot --with-encoders --fine-tune --epochs 1 \
	--allow-intensive --device cuda --batch-size 1 --threads 1 \
	--output PRISM/research/artifacts/pilot_cuda_guarded
```

Do not launch a full three-seed run until this pilot completes and its VRAM/RAM
and elapsed-time logs have been reviewed. To stay hardware-independent, omit
`--device cuda`; CPU remains the default.

16 GB system RAM is a reasonable target, not a certified minimum. Default runs
are limited to 300 PRs total, one CPU thread, CPU-only execution, 128-token
sequences and batch size 2. BLAS/OpenMP/tokenizer parallelism is constrained;
forest workers and CV jobs are 1, with 50-tree forests. Process priority is
lowered on supported Linux hosts. These are thread/workload limits, not a hard
RAM or thermal guarantee. WSL allocation and other applications still matter.

## Run

Output directories must be new or empty for generation and complete runs.
No commands overwrite the tracked legacy CSVs.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest PRISM/research/tests -q
.venv/bin/python -m PRISM.research.cli run-all --output PRISM/research/artifacts/my_safe_run
.venv/bin/python -m PRISM.research.cli status --output PRISM/research/artifacts/my_safe_run
.venv/bin/python -m PRISM.research.cli run-all --resume --output PRISM/research/artifacts/my_safe_run
```

Both `smoke` and `run-all` now default to a 100-per-difficulty lexical-only smoke
profile. Results are explicitly labelled; this is not the full BGE experiment.
`--with-encoders` opts into small MiniLM CPU inference, still within safe limits.
Fine-tuning, pilot/full profiles, larger batches and a second CPU thread require
`--allow-intensive`. They are not run automatically or recommended on a
constrained laptop. The override limits threads to 2 and batches to 4 on CPU;
CUDA further requires the stricter VRAM guards above.

For optional safe CPU inference, after installing encoder dependencies:

```bash
.venv/bin/python -m PRISM.research.cli smoke --with-encoders --output PRISM/research/artifacts/my_small_encoder_run
```

Pilot (1,000 per difficulty) and full (3,000 per difficulty across three seeds)
remain opt-in research profiles for suitably monitored hardware. Full requires
`--with-encoders --allow-intensive`. The dissertation's BGE/fine-tuning results
cannot be replaced by lexical-only or small-MiniLM scores.

Managed runs save `run_state.json` atomically before and after each stage.
`--resume` skips completed stages only for identical settings and validates the
dataset fingerprint. A different model/seed/profile needs a new output directory.
Older partial runs without stage records are not restarted automatically.
Missing files in a completed stage are reported rather than silently skipped.

The verified recovery run on this laptop took 2.45 seconds with about 247 MiB
peak process RAM and no GPU computation. That is one measured small run, not a
guarantee for another machine. The earlier restart's cause was not established;
the intensive pilot was not restarted. The former thread settings allowed more
parallel work than is appropriate for this default laptop workflow.

Individual stages support `generate`, `features`, `finetune`, `train-alignment`,
`train-risk`, `evaluate`, and `report`. Use `--help` for their arguments. Stage
commands operate on an existing generated dataset; specify the same seed as the
dataset when invoking them manually. Standalone `finetune` exports a checkpoint;
`features --fine-tune` performs the integrated trained-feature experiment.

To continue a benchmark that already has its own `bge_finetuned/` checkpoint and
`finetuning.json`, use `features --with-encoders --reuse-finetuned` instead of
`--fine-tune`. This keeps the checkpoint unchanged and computes both frozen and
fine-tuned comparison features. It cannot be combined with `--fine-tune` and is
only available for the `features` stage. Then run `train-alignment`, `train-risk`,
and `evaluate` against the same directory. Keep seed/config settings unchanged.

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m PRISM.research.cli features \
	--output PRISM/research/artifacts/cpu_bge_small_seed42 \
	--with-encoders --reuse-finetuned --allow-intensive \
	--device cpu --threads 1 --batch-size 1
```

Measured on this laptop, the small CPU BGE chunk trained for one epoch in 3:40
with about 3.03 GiB peak process RAM. Its remaining feature extraction took 6:06
with about 1.49 GiB peak RAM, without repeating fine-tuning. Classifiers and
evaluation then completed on the same 300-record benchmark. CUDA was hidden;
these measurements are not a full multi-seed runtime estimate or a guarantee for
another laptop. The smoke comparison selected frozen-BGE lexical features rather
than fine-tuned features; do not assume fine-tuning always improves performance.

## Results and Inference

Each run contains `report.html`, `calibration.png`, `predictions.csv`,
`metrics.json`, split manifests, a SHA-256 dataset fingerprint, feature metadata,
calibration/threshold-aware model artifacts, and forest explanations.
Open `report.html` directly in a browser; no web server is required.
Full runs, if explicitly launched, have one report per seed and an AP mean/std
summary. Safe run metadata records the actual native thread-pool limits.

```bash
.venv/bin/python -m PRISM.research.cli predict \
	--run PRISM/research/artifacts/my_encoder_smoke \
	--input new_pull_requests.csv --jira new_jira_tickets.csv \
	--output PRISM/research/artifacts/new_predictions
```

Inference needs no target labels, split fields, difficulty, or latent values.
PR input requires `pr_id`, `jira_id`, `title`, `description`, `code_change_summary`,
`branch`, `files_changed` as a JSON list, and the eight observable fields listed in
`research/data.py:RISK_FEATURES`. Jira input requires `id`, `title`, `description`,
and `module`. All referenced issues must exist. Activity/comments must represent
information available at the intended review time. Use only trusted joblib/model
artifacts: loading arbitrary serialized files can execute code.

## Experimental Protocol

- Canonical module/task groups are disjoint across chronological train (60%),
	selection (10%), calibration (5%), threshold selection (5%), and test (20%).
- Risk compares class-weighted Random Forest, logistic regression, gradient
	boosting, and a dummy model; preprocessing is refitted within training-only
	stratified group CV. Candidate selection uses development AP, never test AP.
- Alignment compares lexical-only, frozen BGE semantic/lexical/hybrid variants,
	fine-tuned variants when requested, and description-only MiniLM with fixed 0.7
	cosine threshold. Lexical IDF/vocabularies use training text only.
- Safe mode defaults to lexical-only; optional safe encoder inference uses MiniLM
	instead of BGE and never fine-tunes. Model names and revisions are recorded so
	smaller-model results cannot be mistaken for the report-required BGE experiment.
- BGE uses normalized, symmetric text embeddings without generative prompts.
	CPU is the default; optional CUDA requires explicit opt-in and VRAM preflight.
	Cross-encoder scores are features, not probabilities. Mining uses training-only
	candidates and excludes every positive group in the current batch. Fine-tuning
	uses at most one positive example per task group per epoch; this limits repeated
	templates, but also limits training diversity. Checkpoints use selection data.
- Sigmoid calibration and F1 threshold selection use separate development sets.
	Test classification, retrieval, clean-label audit, difficulty/type breakdowns,
	and group-bootstrap intervals are reported separately.
- PR-AUC is summarized with average precision (AP). Aligned-positive and
	misaligned-positive results are separate. Single-class AUCs are undefined.
- Retrieval uses split-local, time-available candidate tickets and actual relevant
	issues, not intentionally incorrect linked IDs. Vague/no-positive queries are
	excluded with their count reported. Retrieval is the bi-encoder stage, not the
	calibrated classifier. Literal issue IDs are not semantic/lexical signals.
- TreeSHAP explains the uncalibrated Random Forest baseline. It does not explain
	the selected model's calibrated probability and is not causal. Additivity is
	checked against the forest output before explanations are saved.

## Requirements to Evidence

| Report requirement | Implementation | Verification/artifact |
| --- | --- | --- |
| Seeded difficult synthetic data and hidden labels | `research/data.py` | `test_data.py`, dataset/audit/split manifests |
| Multi-field BGE, BM25, TF-IDF, structural signals | `research/alignment.py` | `test_alignment.py`, features/revisions |
| Cross-encoder and hard-negative MNRL fine-tuning | `research/alignment.py` | Real-model smoke, `finetuning.json`, checkpoint |
| Calibrated alignment and chosen thresholds | `research/modeling.py` | Alignment artifacts and ablations |
| Risk baselines, group CV, SHAP | `research/risk.py` | `test_models.py`, baseline metrics/explanations |
| Held-out metrics, rankings, ablations, false-positive/negative analysis | `research/evaluation.py` | `metrics.json`, error counts and examples by difficulty/type |
| Per-test-PR explanations and reviewer output | `research/reporting.py` | `predictions.csv` covers every held-out PR with top forest contributions |
| Unseen-input probabilities and explanations | `research/cli.py predict` | `predictions.csv` explains each supplied PR |
| Reviewer prototype and limits | `research/reporting.py` | `test_evaluation.py`, HTML/CSV/image output |

## Limitations

Read `research/DATASET_CARD.md` before interpreting results. Tiny smoke scores
verify execution, not model quality; even perfect alignment on synthetic text
does not establish transfer to real projects. Risk labels are assumed, noisy
latent outcomes, not empirical defects. No production certification is claimed.
The full multi-seed experiment must be run before drawing dissertation conclusions.

The report's final administrative page names Predictive CI/CD, unlike the PRISM
scope on pages 1-9. PRISM is the user-confirmed specification; the PDF is unchanged.
Generative LLMs, organizational integrations, executable-code analysis, and
production deployment are out of scope.
