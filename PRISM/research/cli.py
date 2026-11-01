"""Command-line entry point; no model downloads occur on import."""

import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
from threadpoolctl import threadpool_info, threadpool_limits

from .data import RISK_FEATURES, generate_benchmark, load_benchmark, write_benchmark
from .resources import check_record_budget, configure_torch, preflight_cuda, prepare_environment, validate_request


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def config_for(args):
    config = json.loads(Path(getattr(args, "config", DEFAULT_CONFIG)).read_text(encoding="utf-8"))
    if not getattr(args, "allow_intensive", False):
        config["bi_encoder"] = config.get("safe_bi_encoder", "sentence-transformers/all-MiniLM-L6-v2")
        config["max_sequence_length"] = min(config["max_sequence_length"], 128)
        config["profiles"]["smoke"] = min(config["profiles"]["smoke"], 100)
    return config


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def generate(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not getattr(args, "_managed_run", False):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {output}")
    rows, tickets, audit = generate_benchmark(args.records, args.seed)
    write_benchmark(output, rows, tickets, audit, {"records_per_difficulty": args.records, "seed": args.seed})
    print(json.dumps({"stage": "generate", "records": len(rows), "output": str(output)}, indent=2))


def features(args):
    import gc
    import joblib
    import numpy as np
    from .alignment import AlignmentFeatures, BiEncoder, finetune, issue_text, load_cross_encoder, pr_text
    from .evaluation import rank_queries

    output = Path(args.output)
    rows, tickets, audit = load_benchmark(output)
    check_record_budget(len(rows), getattr(args, "allow_intensive", False))
    reuse_checkpoint = getattr(args, "reuse_finetuned", False)
    checkpoint = output / "bge_finetuned"
    if reuse_checkpoint and (not checkpoint.is_dir() or not (output / "finetuning.json").is_file()):
        raise ValueError("Checkpoint reuse requires bge_finetuned/ and finetuning.json in this benchmark directory")
    config = config_for(args)
    np.random.seed(args.seed)
    extractor = AlignmentFeatures().fit(rows, tickets)
    lexical, lexical_names = extractor.transform(rows, tickets)
    arrays = {"lexical": lexical}
    names = {"lexical": lexical_names}
    metadata = {"mode": "lexical-only; no encoder or fine-tuning results", "feature_names": names,
                "python": platform.python_version(), "seed": args.seed, "config": config,
                "runtime": {"device": args.device, "batch_size": args.batch_size, "epochs": args.epochs,
                            "fine_tune": args.fine_tune, "cpu_threads": args.threads,
                            "safe_mode": not args.allow_intensive,
                            "reused_finetuned_checkpoint": reuse_checkpoint},
                "packages": {name: importlib.metadata.version(name) for name in (
                    "numpy", "pandas", "scikit-learn", "rank-bm25", "shap", "matplotlib", "joblib", "jinja2",
                )}}
    if not args.lexical_only:
        metadata["packages"].update({name: importlib.metadata.version(name)
                                      for name in ("torch", "sentence-transformers", "transformers")})
        import torch
        torch.manual_seed(args.seed)
        configure_torch()
        purpose = "training" if args.fine_tune else "inference"
        encoder = BiEncoder(config["bi_encoder"], args.batch_size, config["max_sequence_length"],
                    args.device, purpose=purpose)
        metadata["runtime"]["gpu_preflight"] = encoder.gpu_preflight
        metadata["runtime"]["gpu_required"] = args.device == "cuda"
        metadata["bi_encoder_revision"] = encoder.model[0].auto_model.config._commit_hash
        frozen, frozen_names = extractor.transform(rows, tickets, encoder)
        arrays["encoder_lexical"], names["encoder_lexical"] = frozen, frozen_names
        semantic_indices = [index for index, name in enumerate(frozen_names) if name.startswith("semantic_")]
        arrays["encoder_semantic"] = frozen[:, semantic_indices]
        names["encoder_semantic"] = [frozen_names[index] for index in semantic_indices]
        arrays["ranks"] = np.array([-1 if rank is None else rank for rank in rank_queries(encoder, rows, tickets, audit)])
        if args.fine_tune or reuse_checkpoint:
            if not reuse_checkpoint:
                history = finetune(encoder, rows, tickets, checkpoint, epochs=args.epochs, seed=args.seed)
                save_json(output / "finetuning.json", history)
            else:
                print(f"Reusing saved checkpoint: {checkpoint}; no BGE retraining", flush=True)
            encoder.unload()
            del encoder
            encoder = BiEncoder(str(checkpoint), args.batch_size, config["max_sequence_length"], args.device)
            tuned, tuned_names = extractor.transform(rows, tickets, encoder)
            arrays["finetuned_lexical"], names["finetuned_lexical"] = tuned, tuned_names
            arrays["finetuned_ranks"] = np.array([-1 if rank is None else rank for rank in rank_queries(encoder, rows, tickets, audit)])
        encoder.unload()
        del encoder
        gc.collect()
        cross_encoder = load_cross_encoder(config["cross_encoder"], config["max_sequence_length"], args.device)
        metadata["cross_encoder_revision"] = cross_encoder.model.config._commit_hash
        lookup = {ticket["id"]: ticket for ticket in tickets}
        pairs = [(issue_text(lookup[row["jira_id"]]), pr_text(row)) for row in rows]
        scores = cross_encoder.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
        arrays["encoder_hybrid"] = np.column_stack((frozen, scores))
        names["encoder_hybrid"] = frozen_names + ["cross_encoder_score"]
        if args.fine_tune or reuse_checkpoint:
            arrays["finetuned_hybrid"] = np.column_stack((arrays["finetuned_lexical"], scores))
            names["finetuned_hybrid"] = names["finetuned_lexical"] + ["cross_encoder_score"]
        del cross_encoder
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        legacy = BiEncoder("sentence-transformers/all-MiniLM-L6-v2", args.batch_size,
                           config["max_sequence_length"], args.device)
        metadata["legacy_encoder_revision"] = legacy.model[0].auto_model.config._commit_hash
        jira_embeddings = legacy.encode([lookup[row["jira_id"]]["description"] for row in rows])
        pr_embeddings = legacy.encode([row["description"] for row in rows])
        arrays["legacy_cosine"] = np.einsum("ij,ij->i", jira_embeddings, pr_embeddings)
        legacy.unload()
        metadata["mode"] = "encoder + lexical + cross-encoder" + (
            "; reused fine-tuned checkpoint" if reuse_checkpoint
            else " + fine-tuning" if args.fine_tune else "; frozen encoders")
    joblib.dump(extractor, output / "feature_extractor.joblib")
    np.savez_compressed(output / "features.npz", **arrays)
    metadata["runtime"]["native_thread_pools"] = [{"library": item["internal_api"],
                                                   "threads": item["num_threads"]} for item in threadpool_info()]
    save_json(output / "features.json", metadata)
    print(f"Features saved: {output}; mode: {metadata['mode']}")


def train_alignment_stage(args):
    import joblib
    import numpy as np
    from .alignment import train_alignment

    output = Path(args.output)
    rows, _, _ = load_benchmark(output)
    check_record_budget(len(rows), getattr(args, "allow_intensive", False))
    metadata = json.loads((output / "features.json").read_text())
    bundles = {}
    with np.load(output / "features.npz") as arrays:
        for name, names in metadata["feature_names"].items():
            bundle, _ = train_alignment(arrays[name], rows, names, seed=args.seed)
            bundles[name] = bundle
    selected = max(bundles, key=lambda name: bundles[name].comparisons[bundles[name].name]["selection_ap"])
    metadata["selected_feature_set"] = selected
    joblib.dump(bundles, output / "alignment_variants.joblib")
    joblib.dump(bundles[selected], output / "alignment.joblib")
    save_json(output / "features.json", metadata)
    print(f"Alignment trained: {selected}/{bundles[selected].name}")


def train_risk_stage(args):
    import joblib
    from .risk import train_risk

    output = Path(args.output)
    rows, _, _ = load_benchmark(output)
    check_record_budget(len(rows), getattr(args, "allow_intensive", False))
    _, bundle, baselines = train_risk(rows, args.seed)
    joblib.dump(bundle, output / "risk.joblib")
    joblib.dump(baselines, output / "risk_baselines.joblib")
    print(f"Risk trained: {bundle.name}")


def evaluate_stage(args):
    import joblib
    import numpy as np
    from .evaluation import binary_metrics, evaluate
    from .reporting import write_report
    from .risk import FEATURE_NAMES, explain_forest, risk_features

    output = Path(args.output)
    rows, _, audit = load_benchmark(output)
    check_record_budget(len(rows), getattr(args, "allow_intensive", False))
    metadata = json.loads((output / "features.json").read_text())
    alignment_model = joblib.load(output / "alignment.joblib")
    risk_model = joblib.load(output / "risk.joblib")
    variants = joblib.load(output / "alignment_variants.joblib")
    baselines = joblib.load(output / "risk_baselines.joblib")
    risk_values = risk_features(rows)
    risk_probability = risk_model.predict_probability(risk_values)
    test = np.array([row["split"] == "test" for row in rows])
    with np.load(output / "features.npz") as arrays:
        selected = metadata["selected_feature_set"]
        alignment_probability = 1 - alignment_model.predict_probability(arrays[selected])
        rank_key = "finetuned_ranks" if selected.startswith("finetuned") else "ranks"
        ranks = [None if rank < 0 else int(rank) for rank in arrays[rank_key]] if rank_key in arrays else None
        metrics = evaluate(rows, audit, alignment_probability, risk_probability,
                           alignment_model.threshold, risk_model.threshold, ranks, args.seed)
        metrics["alignment_ablations"] = {}
        for name, bundle in variants.items():
            probability = bundle.predict_probability(arrays[name])
            metrics["alignment_ablations"][name] = {
                "model": bundle.name, "selection": bundle.comparisons,
                "calibrated_test": binary_metrics([1 - row["is_aligned"] for row in rows if row["split"] == "test"],
                                                   probability[test], bundle.threshold),
                "uncalibrated_test": binary_metrics([1 - row["is_aligned"] for row in rows if row["split"] == "test"],
                                                     bundle.estimator.predict_proba(arrays[name][test])[:, 1]),
            }
        if "legacy_cosine" in arrays:
            scores = (1 - arrays["legacy_cosine"][test]) / 2
            legacy = binary_metrics([1 - row["is_aligned"] for row in rows if row["split"] == "test"], scores, 0.15)
            legacy["brier"] = None
            legacy["score_kind"] = "Remapped cosine score; fixed cosine threshold 0.7; not a probability"
            metrics["alignment_ablations"]["legacy_minilm"] = legacy
    metrics["risk_model_selection"] = risk_model.comparisons
    metrics["risk_baselines_uncalibrated"] = {
        name: binary_metrics([row["is_high_risk"] for row in rows if row["split"] == "test"],
                             model.predict_proba(risk_values[test])[:, 1]) for name, model in baselines.items()
    }
    train = np.array([row["split"] == "train" for row in rows])
    test_indices = np.flatnonzero(test)
    contributions = explain_forest(baselines["random_forest"], risk_values[train],
                                   risk_values[test], FEATURE_NAMES, batch_size=32)
    explanations = {rows[index]["pr_id"]: contribution
                    for index, contribution in zip(test_indices, contributions)}
    save_json(output / "forest_explanations.json", explanations)
    write_report(output, rows, alignment_probability, risk_probability, metrics,
                 alignment_model, risk_model, explanations, metadata["mode"], args.seed)
    print(json.dumps({"test_records": int(test.sum()), "report": str(output / "report.html"),
                      "risk_ap": metrics["overall"]["risk"]["average_precision"],
                      "misalignment_ap": metrics["overall"]["alignment_misaligned_positive"]["average_precision"]}, indent=2))


@threadpool_limits.wrap(limits=1)
def run_pipeline(args):
    output = Path(args.output)
    spec = {"seed": args.seed, "records": args.records, "lexical_only": args.lexical_only,
            "fine_tune": args.fine_tune, "epochs": args.epochs, "config": config_for(args),
            "device": args.device, "batch_size": args.batch_size, "threads": args.threads}
    state_path = output / "run_state.json"
    if args.resume:
        if not state_path.exists():
            raise ValueError("No resumable stage record exists; partial legacy pilot artifacts are preserved. Use a new small output directory")
        state = json.loads(state_path.read_text())
        if state["spec"] != spec:
            raise ValueError("Resume settings differ from the original run; choose a new output directory")
        if "generate" in state["completed"]:
            rows, _, _ = load_benchmark(output)
            check_record_budget(len(rows), args.allow_intensive)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; use --resume for a matching managed run")
        output.mkdir(parents=True, exist_ok=True)
        state = {"spec": spec, "completed": [], "current_stage": None, "last_recorded_status": "ready"}
    args._managed_run = True
    required_outputs = {
        "generate": ("dataset.json", "pull_requests.csv", "jira_tickets.csv", "audit.json", "splits.json"),
        "features": ("features.json", "features.npz", "feature_extractor.joblib"),
        "train-alignment": ("alignment.joblib", "alignment_variants.joblib"),
        "train-risk": ("risk.joblib", "risk_baselines.joblib"),
        "evaluate": ("metrics.json", "report.html", "predictions.csv", "calibration.png"),
    }
    for name, action in (("generate", generate), ("features", features),
                         ("train-alignment", train_alignment_stage), ("train-risk", train_risk_stage),
                         ("evaluate", evaluate_stage)):
        if name in state["completed"]:
            missing = [filename for filename in required_outputs[name] if not (output / filename).is_file()]
            if missing:
                raise ValueError(f"Completed stage {name} is missing artifacts: {missing}; rerun that stage explicitly")
            print(f"Resume: keeping completed stage {name}", flush=True)
            continue
        state.update(current_stage=name, last_recorded_status="started")
        save_json(state_path, state)
        print(f"Starting {name} (CPU threads: {args.threads}; device: {args.device})", flush=True)
        try:
            action(args)
        except Exception as error:
            state.update(last_recorded_status="failed", error=str(error))
            save_json(state_path, state)
            raise
        state["completed"].append(name)
        state.update(last_recorded_status="complete")
        state.pop("error", None)
        save_json(state_path, state)


def run_all(args):
    import copy
    import numpy as np

    config = config_for(args)
    args.records = config["profiles"][args.profile]
    if args.profile == "full" and args.lexical_only:
        raise ValueError("The full research profile requires real encoders; use smoke/pilot for lexical-only runs")
    if args.profile == "full":
        root = Path(args.output)
        if root.exists() and any(root.iterdir()) and not args.resume:
            raise ValueError("Full-run output directory must be new or empty")
        results = []
        for seed in config["seeds"]:
            child = copy.copy(args)
            child.output, child.seed, child.fine_tune = str(root / f"seed_{seed}"), seed, True
            child.resume = args.resume and (Path(child.output) / "run_state.json").is_file()
            run_pipeline(child)
            results.append(json.loads((Path(child.output) / "metrics.json").read_text()))
        summary = {}
        for task in ("risk", "alignment_misaligned_positive"):
            values = [result["overall"][task]["average_precision"] for result in results]
            summary[task] = {"ap_mean": float(np.mean(values)), "ap_std": float(np.std(values)), "seeds": config["seeds"]}
        save_json(root / "seed_summary.json", summary)
    else:
        run_pipeline(args)


def status_stage(args):
    output = Path(args.output)
    state_path = output / "run_state.json"
    if state_path.exists():
        print(json.dumps(json.loads(state_path.read_text()), indent=2))
    else:
        print(json.dumps({"managed_run": False, "partial_artifacts": sorted(path.name for path in output.iterdir()),
                          "note": "Artifact presence does not prove a completed or valid run; no automatic restart"}, indent=2))


def finetune_stage(args):
    from .alignment import BiEncoder, finetune
    output = Path(args.output)
    rows, tickets, _ = load_benchmark(output)
    config = config_for(args)
    checkpoint = output / "bge_finetuned"
    if checkpoint.exists():
        raise ValueError("Fine-tuned checkpoint already exists; choose a new run directory")
    encoder = BiEncoder(config["bi_encoder"], args.batch_size, config["max_sequence_length"],
                        args.device, purpose="training")
    history = finetune(encoder, rows, tickets, checkpoint, args.epochs, args.seed)
    encoder.unload()
    save_json(output / "finetuning.json", history)
    print(f"Fine-tuned checkpoint: {checkpoint}")


def predict_stage(args):
    import gc
    import joblib
    import numpy as np
    from .alignment import BiEncoder, issue_text, load_cross_encoder, pr_text
    from .risk import risk_features

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prediction output directory must be new or empty")
    run = Path(args.run)
    metadata = json.loads((run / "features.json").read_text())
    config = metadata["config"]
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with Path(args.jira).open(newline="", encoding="utf-8") as handle:
        tickets = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Prediction input is empty")
    check_record_budget(len(rows), getattr(args, "allow_intensive", False))
    for row in rows:
        for name in RISK_FEATURES:
            row[name] = int(row[name])
    risk_model = joblib.load(run / "risk.joblib")
    risk_baseline = joblib.load(run / "risk_baselines.joblib")["random_forest"]
    alignment_model = joblib.load(run / "alignment.joblib")
    extractor = joblib.load(run / "feature_extractor.joblib")
    selected = metadata["selected_feature_set"]
    encoder = None
    if selected != "lexical":
        model_name = str(run / "bge_finetuned") if selected.startswith("finetuned") else config["bi_encoder"]
        revision = None if selected.startswith("finetuned") else metadata["bi_encoder_revision"]
        encoder = BiEncoder(model_name, args.batch_size, config["max_sequence_length"], args.device, revision)
    values, names = extractor.transform(rows, tickets, encoder)
    if encoder is not None:
        encoder.unload()
        del encoder
        gc.collect()
    if "cross_encoder_score" in alignment_model.feature_names:
        cross = load_cross_encoder(config["cross_encoder"], config["max_sequence_length"],
                                   args.device, metadata["cross_encoder_revision"])
        lookup = {ticket["id"]: ticket for ticket in tickets}
        pairs = [(issue_text(lookup[row["jira_id"]]), pr_text(row)) for row in rows]
        values = np.column_stack((values, cross.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)))
        names.append("cross_encoder_score")
    values = values[:, [names.index(name) for name in alignment_model.feature_names]]
    aligned = 1 - alignment_model.predict_probability(values)
    risk_values = risk_features(rows)
    risk = risk_model.predict_probability(risk_values)
    training_rows, _, _ = load_benchmark(run)
    training_features = risk_features([row for row in training_rows if row["split"] == "train"])
    from .risk import FEATURE_NAMES, explain_forest
    contributions = explain_forest(risk_baseline, training_features, risk_values,
                                   FEATURE_NAMES, batch_size=32)
    results = [{"pr_id": row["pr_id"], "jira_id": row["jira_id"],
                "alignment_probability": float(aligned[index]), "risk_probability": float(risk[index]),
                "misaligned": int(1 - aligned[index] >= alignment_model.threshold),
                "high_risk": int(risk[index] >= risk_model.threshold),
                "risk_contributions": json.dumps(sorted(contributions[index].items(),
                                                         key=lambda item: abs(item[1]), reverse=True)[:3])}
               for index, row in enumerate(rows)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Predicted {len(results)} PRs: {output / 'predictions.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PRISM controlled synthetic research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate", help="Generate a labelled benchmark and split manifests")
    generation.add_argument("--records", type=int, default=100, help="PRs per difficulty; minimum 100")
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--output", required=True)
    generation.add_argument("--threads", type=int, default=1)
    generation.add_argument("--allow-intensive", action="store_true")
    generation.set_defaults(action=generate)
    for command, action in (("features", features), ("finetune", finetune_stage), ("train-alignment", train_alignment_stage),
                            ("train-risk", train_risk_stage), ("evaluate", evaluate_stage),
                            ("report", evaluate_stage), ("smoke", run_all), ("run-all", run_all)):
        stage = commands.add_parser(command)
        stage.add_argument("--output", required=True, help="Benchmark/run directory")
        stage.add_argument("--seed", type=int, default=42)
        stage.add_argument("--config", default=str(DEFAULT_CONFIG))
        stage.add_argument("--batch-size", type=int, default=2)
        stage.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        backend = stage.add_mutually_exclusive_group()
        backend.add_argument("--lexical-only", action="store_true", help="Lightweight offline baseline (default)")
        backend.add_argument("--with-encoders", action="store_true", help="Opt in to small CPU encoders in safe mode")
        stage.add_argument("--threads", type=int, default=1)
        stage.add_argument("--allow-intensive", action="store_true", help="Explicitly permit larger runs/fine-tuning; not recommended on constrained laptops")
        stage.add_argument("--resume", action="store_true", help="Continue a matching managed run without repeating completed stages")
        stage.add_argument("--fine-tune", action="store_true")
        if command == "features":
            stage.add_argument("--reuse-finetuned", action="store_true",
                               help="Reuse this benchmark's saved bge_finetuned checkpoint without retraining")
        stage.add_argument("--epochs", type=int, default=1)
        stage.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
        stage.set_defaults(action=action)
    prediction = commands.add_parser("predict", help="Predict unseen linked PRs using trusted saved artifacts")
    prediction.add_argument("--run", required=True)
    prediction.add_argument("--input", required=True, help="PR CSV with text and observable metadata; no labels needed")
    prediction.add_argument("--jira", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--batch-size", type=int, default=2)
    prediction.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    prediction.add_argument("--threads", type=int, default=1)
    prediction.add_argument("--allow-intensive", action="store_true")
    prediction.set_defaults(action=predict_stage)
    status = commands.add_parser("status", help="Read saved stage records without loading models")
    status.add_argument("--output", required=True)
    status.add_argument("--threads", type=int, default=1)
    status.set_defaults(action=status_stage)
    args = parser.parse_args(argv)
    try:
        validate_request(args)
        device = getattr(args, "device", "cpu")
        prepare_environment(args.threads, device)
        if hasattr(args, "with_encoders"):
            args.lexical_only = not args.with_encoders
        if getattr(args, "reuse_finetuned", False):
            if args.lexical_only or args.fine_tune:
                raise ValueError("--reuse-finetuned requires --with-encoders and cannot be combined with --fine-tune")
        if device == "cuda":
            if getattr(args, "lexical_only", False):
                raise ValueError("CUDA was requested for a lexical-only command; add --with-encoders")
            training = (args.command == "finetune" or getattr(args, "fine_tune", False)
                        or (args.command in {"run-all", "smoke"} and getattr(args, "profile", None) == "full"))
            report = preflight_cuda("training" if training else "inference")
            print(f"CUDA preflight passed: {report}; per-process allocator cap is 70% of VRAM", flush=True)
        args.action(args)
    except (ValueError, RuntimeError, FileNotFoundError, ModuleNotFoundError,
            importlib.metadata.PackageNotFoundError) as error:
        parser.exit(2, f"PRISM: {error}\n")


if __name__ == "__main__":
    main()