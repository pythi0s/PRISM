import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PRISM.research.data import (
    RISK_FEATURES, SPLITS, generate_benchmark, load_benchmark, parse_files,
    tokens, validate_benchmark, write_benchmark,
)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.tickets, cls.audit = generate_benchmark(100, 42)

    def test_seed_reproducibility(self):
        self.assertEqual((self.rows, self.tickets, self.audit), generate_benchmark(100, 42))
        self.assertNotEqual(self.rows, generate_benchmark(100, 123)[0])

    def test_jira_pr_text_has_seeded_paraphrase_and_synonym_variation(self):
        from PRISM.research.data import vary_synonyms
        import random
        groups = {}
        for row in self.rows:
            if row["split"] == "train" and row["difficulty"] == "easy":
                groups.setdefault(row["group_id"], set()).add(row["title"])
        self.assertGreaterEqual(max(map(len, groups.values())), 2)
        self.assertNotEqual(vary_synonyms("preserve existing behavior", random.Random(1)),
                            vary_synonyms("preserve existing behavior", random.Random(2)))

    def test_class_support_and_chronology(self):
        for split in SPLITS:
            subset = [row for row in self.rows if row["split"] == split]
            for label in ("is_aligned", "is_high_risk"):
                self.assertEqual({row[label] for row in subset}, {0, 1})
        validate_benchmark(self.rows, self.tickets, self.audit)

    def test_latents_are_not_features(self):
        self.assertTrue(all(set(RISK_FEATURES) <= set(row) for row in self.rows))
        self.assertTrue(all("risk_probability" not in row for row in self.rows))
        self.assertTrue(all(0 < item["risk_probability"] < 1 for item in self.audit))

    def test_file_schema_and_identifiers(self):
        self.assertEqual(parse_files('["src/auth/main.py"]'), ["src/auth/main.py"])
        for malformed in ('{"path": "a"}', '[42]', 'invalid'):
            with self.assertRaises(ValueError):
                parse_files(malformed)
        self.assertEqual(tokens("PROJ-123 Fix auth PR-12"), ["fix", "auth"])

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as output:
            write_benchmark(output, self.rows, self.tickets, self.audit, {"seed": 42})
            self.assertEqual(load_benchmark(output), (self.rows, self.tickets, self.audit))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            generate_benchmark(10)
        bad = [dict(row) for row in self.rows]
        bad[0]["risk_probability"] = 0.7
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_benchmark(bad, self.tickets, self.audit)

    def test_arbitrary_record_count(self):
        self.assertEqual(len(generate_benchmark(103, 42)[0]), 309)

    def test_cli_does_not_overwrite(self):
        from PRISM.research.cli import main
        with tempfile.TemporaryDirectory() as output:
            main(["generate", "--output", output])
            with self.assertRaises(SystemExit) as caught:
                main(["generate", "--output", output])
            self.assertEqual(caught.exception.code, 2)


    def test_manifest_detects_modified_inputs(self):
        with tempfile.TemporaryDirectory() as output:
            write_benchmark(output, self.rows, self.tickets, self.audit, {"seed": 42})
            path = Path(output) / "pull_requests.csv"
            path.write_text(path.read_text().replace("Maintenance", "Changed title", 1))
            with self.assertRaisesRegex(ValueError, "hash"):
                load_benchmark(output)

    def test_safe_mode_blocks_expensive_work_before_action(self):
        from PRISM.research.cli import main
        with patch("PRISM.research.cli.run_all") as action:
            for options in (["--profile", "pilot"], ["--fine-tune"], ["--device", "auto"],
                            ["--threads", "2"], ["--batch-size", "4"]):
                with self.assertRaises(SystemExit):
                    main(["run-all", "--output", "unused", *options])
            action.assert_not_called()

    def test_safe_defaults_and_record_budget(self):
        from PRISM.research.cli import main
        from PRISM.research.resources import check_record_budget, validate_request, validate_vram_capacity
        with patch("PRISM.research.cli.run_all") as action:
            main(["run-all", "--output", "unused"])
            args = action.call_args.args[0]
            self.assertTrue(args.lexical_only)
            self.assertEqual((args.device, args.threads, args.batch_size, args.profile), ("cpu", 1, 2, "smoke"))
        with self.assertRaises(ValueError):
            check_record_budget(3000)
        from types import SimpleNamespace
        intensive_cuda = SimpleNamespace(threads=1, batch_size=1, epochs=1,
                                         allow_intensive=False, device="cuda", command="run-all")
        with self.assertRaisesRegex(ValueError, "CUDA requires"):
            validate_request(intensive_cuda)
        self.assertEqual(validate_vram_capacity(5500, 6144, "training")["minimum_free_mib"], 5000)
        self.assertEqual(validate_vram_capacity(5500, 6143, "training")["total_mib"], 6143)
        with self.assertRaisesRegex(RuntimeError, "No model was loaded"):
            validate_vram_capacity(4999, 6144, "training")
        with self.assertRaisesRegex(RuntimeError, "total VRAM"):
            validate_vram_capacity(3500, 4096, "inference")
        guarded_cuda = SimpleNamespace(threads=1, batch_size=1, epochs=1,
                                       allow_intensive=True, device="cuda", command="run-all")
        validate_request(guarded_cuda)
        guarded_cuda.batch_size = 2
        with self.assertRaisesRegex(ValueError, "batch-size 1"):
            validate_request(guarded_cuda)

    def test_cuda_cli_requires_preflight_before_action(self):
        from PRISM.research.cli import main
        arguments = ["run-all", "--output", "unused", "--with-encoders",
                     "--allow-intensive", "--device", "cuda", "--batch-size", "1", "--fine-tune"]
        with patch("PRISM.research.cli.run_all") as action, \
                patch("PRISM.research.cli.preflight_cuda", return_value={"free_mib": 5500}) as preflight:
            main(arguments)
            preflight.assert_called_once_with("training")
            action.assert_called_once()
        with patch("PRISM.research.cli.run_all") as action, \
                patch("PRISM.research.cli.preflight_cuda", side_effect=RuntimeError("VRAM below guard")):
            with self.assertRaises(SystemExit):
                main(arguments)
            action.assert_not_called()

    def test_resume_preserves_completed_generation(self):
        import json
        import builtins
        from PRISM.research.cli import main
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / "run")
            command = ["run-all", "--output", output]
            with patch("PRISM.research.cli.features", side_effect=ValueError("injected interruption")):
                with self.assertRaises(SystemExit):
                    main(command)
            state = json.loads((Path(output) / "run_state.json").read_text())
            self.assertEqual(state["completed"], ["generate"])
            self.assertEqual(state["last_recorded_status"], "failed")
            original = (Path(output) / "pull_requests.csv").read_bytes()
            original_import = builtins.__import__

            def no_transformer_import(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "sentence_transformers", "transformers"}:
                    raise AssertionError(f"Default run must not import {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=no_transformer_import):
                main([*command, "--resume"])
            self.assertEqual(original, (Path(output) / "pull_requests.csv").read_bytes())
            state = json.loads((Path(output) / "run_state.json").read_text())
            self.assertEqual(len(state["completed"]), 5)
            (Path(output) / "metrics.json").unlink()
            with self.assertRaises(SystemExit):
                main([*command, "--resume"])

    def test_full_resume_continues_existing_seeds_and_starts_missing_seeds(self):
        import json
        from PRISM.research.cli import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = root / "seed_42"
            previous.mkdir()
            (previous / "run_state.json").write_text("{}")
            observed = []

            def fake_pipeline(args):
                observed.append((args.seed, args.resume, args.records, args.fine_tune))
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                (output / "metrics.json").write_text(json.dumps({"overall": {
                    "risk": {"average_precision": 0.5},
                    "alignment_misaligned_positive": {"average_precision": 0.7},
                }}))

            with patch("PRISM.research.cli.run_pipeline", side_effect=fake_pipeline):
                main(["run-all", "--output", str(root), "--profile", "full", "--resume",
                      "--with-encoders", "--allow-intensive", "--device", "cpu", "--batch-size", "1"])
            self.assertEqual(observed, [(42, True, 3000, True), (123, False, 3000, True),
                                       (2026, False, 3000, True)])
            self.assertTrue((root / "seed_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()