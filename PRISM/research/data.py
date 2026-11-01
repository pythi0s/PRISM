"""Seeded synthetic data with isolated audit labels and chronological splits."""

import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


SPLITS = ("train", "selection", "calibration", "threshold", "test")
FRACTIONS = (0.60, 0.10, 0.05, 0.05, 0.20)
DIFFICULTIES = {
    "easy": {"partial": 0.10, "text_noise": 0.02, "label_noise": 0.00},
    "medium": {"partial": 0.45, "text_noise": 0.10, "label_noise": 0.02},
    "hard": {"partial": 0.75, "text_noise": 0.25, "label_noise": 0.05},
}
MODULES = ("auth", "database", "email", "search", "billing", "storage")
SYNONYMS = {
    "service": ("service", "component", "subsystem"),
    "change": ("change", "update", "modification"),
    "preserve": ("preserve", "retain", "maintain"),
    "existing": ("existing", "current", "previous"),
    "behavior": ("behavior", "functionality", "operation"),
    "request": ("request", "call", "operation"),
    "compatible": ("compatible", "consistent", "backward compatible"),
    "coverage": ("coverage", "tests", "checks"),
}
OPERATIONS = (
    ("refresh expired credentials", "renew stale access tokens"),
    ("validate incoming requests", "check submitted payloads"),
    ("retry failed transactions", "repeat unsuccessful operations"),
    ("remove duplicate records", "deduplicate repeated entries"),
    ("paginate large result sets", "return long collections in pages"),
    ("cache frequent lookups", "reuse commonly requested results"),
    ("sanitize untrusted input", "clean externally supplied values"),
    ("export historical activity", "download past event records"),
    ("schedule delayed processing", "queue deferred background work"),
    ("notify administrators of errors", "alert operators about failures"),
    ("enforce per-user quotas", "limit individual resource consumption"),
    ("restore deleted resources", "recover previously removed objects"),
    ("rotate encryption keys", "replace cryptographic secrets safely"),
    ("cancel pending operations", "stop unfinished requests"),
    ("verify resource ownership", "confirm who controls each object"),
    ("compress archived content", "reduce stored historical data size"),
    ("merge concurrent updates", "reconcile simultaneous modifications"),
    ("expire inactive sessions", "close connections with no recent activity"),
    ("audit privileged actions", "record sensitive administrator changes"),
    ("route regional traffic", "direct requests to nearby regions"),
)
RISK_FEATURES = (
    "lines_added", "lines_deleted", "files_count", "critical_file_count",
    "comment_count", "prior_author_prs", "days_since_author_pr", "test_files_count",
)


def parse_files(value):
    files = json.loads(value) if isinstance(value, str) else value
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise ValueError("files_changed must be a JSON array of paths")
    return files


def tokens(text):
    import re
    return re.findall(r"[a-z]+", re.sub(r"\b(?:PROJ|PR)-\d+\b", "", str(text), flags=re.I).lower())


def vary_synonyms(text, rng):
    import re
    pattern = re.compile(r"\b(" + "|".join(SYNONYMS) + r")\b", re.IGNORECASE)

    def replace(match):
        replacement = rng.choice(SYNONYMS[match.group().lower()])
        return replacement.capitalize() if match.group().istitle() else replacement

    return pattern.sub(replace, text)


def generate_benchmark(records=100, seed=42):
    if records < 100:
        raise ValueError("Use at least 100 records per difficulty for split support")
    rng = random.Random(seed)
    catalog = [(module, operation, paraphrase) for module in MODULES
               for operation, paraphrase in OPERATIONS]
    rng.shuffle(catalog)
    allocations = (72, 12, 6, 6, 24)
    task_partitions = {}
    offset = 0
    for split, count in zip(SPLITS, allocations):
        task_partitions[split] = catalog[offset:offset + count]
        offset += count
    rows, tickets, audit = [], [], []
    author_events = {}
    serial = 0
    for split_index, (split, fraction) in enumerate(zip(SPLITS, FRACTIONS)):
        start = datetime(2026, 1, 1) + timedelta(days=split_index * 40)
        for difficulty, settings in DIFFICULTIES.items():
            split_tasks = task_partitions[split]
            issue_map = {}
            for task_index, (module, operation, paraphrase) in enumerate(split_tasks):
                issue_id = f"PROJ-{seed}-{split}-{difficulty}-{task_index}"
                issue_map[(module, operation)] = issue_id
                jira_title = rng.choice([
                    f"{module}: {operation}", f"{operation.capitalize()} in {module}",
                    f"Request: {paraphrase} for {module}",
                ])
                jira_description = rng.choice([
                    f"The {module} service must {operation}. Preserve existing behavior.",
                    f"Please {paraphrase} in {module}. Keep the current interface compatible.",
                    f"Users need the {module} component to {operation}; add coverage for failures.",
                ])
                tickets.append({
                    "id": issue_id, "title": vary_synonyms(jira_title, rng),
                    "description": vary_synonyms(jira_description, rng),
                    "module": module, "type": "Improvement", "priority": "Medium",
                    "created_date": start.isoformat(), "split": split, "difficulty": difficulty,
                    "group_id": f"{module}:{operation}",
                })
            count = (records - sum(round(records * previous) for previous in FRACTIONS[:-1])
                     if split == "test" else round(records * fraction))
            for record_index in range(count):
                serial += 1
                module, operation, paraphrase = rng.choice(split_tasks)
                true_issue = issue_map[(module, operation)]
                linked_issue = true_issue
                kind = "none"
                if rng.random() < 0.35:
                    draw = rng.random()
                    if draw < settings["partial"]:
                        candidates = [task for task in split_tasks
                                      if task[0] == module and task[1] != operation]
                        kind = "partial"
                    elif draw < 0.90:
                        candidates = [task for task in split_tasks if task[0] != module]
                        kind = "unrelated"
                    else:
                        candidates = []
                        kind = "vague"
                    if candidates:
                        other = rng.choice(candidates)
                        linked_issue = issue_map[(other[0], other[1])]
                    elif kind != "vague":
                        kind = "vague"
                title = rng.choice([
                    f"{module}: {paraphrase}", f"{paraphrase.capitalize()} for {module}",
                    f"{module} change: {operation}",
                ])
                description = rng.choice([
                    f"Updated {module} to {paraphrase}. Existing interfaces remain compatible.",
                    f"This change will {paraphrase} in {module}; callers keep the same contract.",
                    f"For {module}, we now {paraphrase}. Added coverage for the new behavior.",
                ])
                title = vary_synonyms(title, rng)
                description = vary_synonyms(description, rng)
                summary = f"Adjust {module} handlers to {paraphrase}"
                if kind == "vague":
                    title, description, summary = "Maintenance", "Minor cleanup", "Refactor helpers"
                if rng.random() < settings["text_noise"]:
                    description = rng.choice(["", description.replace("service", "servce"),
                                              description + " Also updated unrelated documentation."])
                complexity, familiarity, propensity = (rng.gauss(0, 1) for _ in range(3))
                probability = 1 / (1 + math.exp(-(-1 + 1.2 * complexity - 0.7 * familiarity
                                                + 0.8 * propensity + rng.gauss(0, 0.4))))
                clean_risk = int(rng.random() < probability)
                created = start + timedelta(days=1 + record_index * 25 / max(count, 1))
                author = f"author-{rng.randrange(12)}"
                history = author_events.get(author, [])
                prior = [event for event in history if event < created]
                author_events.setdefault(author, []).append(created)
                files_count = max(1, min(10, round(3 + complexity + rng.gauss(0, 1))))
                files = [f"src/{module}/handler_{file_index}.py" for file_index in range(files_count)]
                if rng.random() < 0.65:
                    files.append(f"tests/test_{module}.py")
                aligned = int(kind == "none")
                rows.append({
                    "pr_id": f"PR-{seed}-{serial}", "jira_id": linked_issue,
                    "title": title, "description": description, "code_change_summary": summary,
                    "branch": f"change/{module}", "files_changed": json.dumps(files),
                    "created_date": created.isoformat(), "author": author,
                    "lines_added": max(1, round(80 + 35 * complexity + rng.gauss(0, 30))),
                    "lines_deleted": max(0, round(30 + 15 * complexity + rng.gauss(0, 15))),
                    "files_count": len(files),
                    "critical_file_count": sum("/auth/" in path or "/database/" in path for path in files),
                    "test_files_count": sum(path.startswith("tests/") for path in files),
                    "comment_count": max(0, round(4 - familiarity + rng.gauss(0, 2))),
                    "prior_author_prs": len(prior),
                    "days_since_author_pr": (created - max(prior)).days if prior else 30,
                    "is_aligned": aligned ^ int(rng.random() < settings["label_noise"]),
                    "is_high_risk": clean_risk ^ int(rng.random() < settings["label_noise"]),
                    "split": split, "difficulty": difficulty,
                    "misalignment_type": kind, "group_id": f"{module}:{operation}",
                })
                audit.append({"pr_id": rows[-1]["pr_id"], "clean_is_aligned": aligned,
                              "clean_is_high_risk": clean_risk, "latent_complexity": complexity,
                              "latent_familiarity": familiarity, "latent_propensity": propensity,
                              "risk_probability": probability,
                              "relevant_jira_ids": [] if kind == "vague" else [true_issue]})
    validate_benchmark(rows, tickets, audit)
    return rows, tickets, audit


def validate_benchmark(rows, tickets, audit):
    if not rows:
        raise ValueError("Benchmark is empty")
    if len({row["pr_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate PR IDs")
    issues = {ticket["id"]: ticket for ticket in tickets}
    if (len(issues) != len(tickets) or len(audit) != len(rows)
            or {item["pr_id"] for item in audit} != {row["pr_id"] for row in rows}):
        raise ValueError("Invalid ticket or audit identities")
    groups = {}
    for ticket in tickets:
        previous = groups.setdefault(ticket["group_id"], ticket["split"])
        if previous != ticket["split"]:
            raise ValueError("Canonical ticket group crosses partitions")
    for row in rows:
        issue = issues.get(row["jira_id"])
        if issue is None or issue["split"] != row["split"] or issue["difficulty"] != row["difficulty"]:
            raise ValueError("Linked issue is unknown or crosses a split/difficulty boundary")
        if issue["created_date"] > row["created_date"]:
            raise ValueError("Issue was not available at review time")
        parse_files(row["files_changed"])
        if any(key.startswith("latent_") or key.startswith("clean_") or key == "risk_probability" for key in row):
            raise ValueError("Audit information leaked into benchmark inputs")
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Canonical task group crosses partitions")
    row_lookup = {row["pr_id"]: row for row in rows}
    for item in audit:
        row = row_lookup[item["pr_id"]]
        for issue_id in item["relevant_jira_ids"]:
            issue = issues.get(issue_id)
            if (issue is None or issue["split"] != row["split"]
                    or issue["difficulty"] != row["difficulty"]
                    or issue["created_date"] > row["created_date"]):
                raise ValueError("Retrieval truth crosses availability or split boundaries")
    for left, right in zip(SPLITS, SPLITS[1:]):
        earlier = [row["created_date"] for row in rows if row["split"] == left]
        later = [row["created_date"] for row in rows if row["split"] == right]
        if not earlier or not later or max(earlier) >= min(later):
            raise ValueError("Partitions must be nonempty and time ordered")
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        for label in ("is_aligned", "is_high_risk"):
            if {row[label] for row in subset} != {0, 1}:
                raise ValueError(f"{split}/{label} needs both classes; increase records or change the logged seed")


def write_benchmark(output, rows, tickets, audit, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("pull_requests.csv", rows), ("jira_tickets.csv", tickets)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {split: [row["pr_id"] for row in rows if row["split"] == split] for split in SPLITS}
    (output / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    payload = json.dumps({"rows": rows, "tickets": tickets, "audit": audit}, sort_keys=True)
    metadata = {"config": config, "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "records": len(rows), "difficulty_profiles": DIFFICULTIES,
                "split_counts": {split: len(ids) for split, ids in manifest.items()},
                "realized_misalignment_counts": {difficulty: dict(Counter(
                    row["misalignment_type"] for row in rows if row["difficulty"] == difficulty
                )) for difficulty in DIFFICULTIES}}
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_benchmark(output):
    output = Path(output)
    with (output / "pull_requests.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (*RISK_FEATURES, "is_aligned", "is_high_risk"):
            row[key] = int(row[key])
    with (output / "jira_tickets.csv").open(newline="", encoding="utf-8") as handle:
        tickets = list(csv.DictReader(handle))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    validate_benchmark(rows, tickets, audit)
    payload = json.dumps({"rows": rows, "tickets": tickets, "audit": audit}, sort_keys=True)
    recorded = json.loads((output / "dataset.json").read_text(encoding="utf-8"))["sha256"]
    if hashlib.sha256(payload.encode()).hexdigest() != recorded:
        raise ValueError("Benchmark hash differs from its generation manifest")
    return rows, tickets, audit