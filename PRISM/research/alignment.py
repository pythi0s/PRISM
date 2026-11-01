"""Multi-field encoder alignment with training-only lexical statistics."""

import copy
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import parse_files, tokens
from .modeling import train_classifiers


FIELDS = ("title", "description", "code_change_summary", "files_changed")
LEXICAL_NAMES = ["tfidf_cosine", "bm25", "identifier_overlap", "module_path_match",
                 "branch_overlap", "missing_description"]


def issue_text(ticket):
    return f'{ticket["title"]}. {ticket["description"]}'


def pr_fields(row):
    return [str(row.get(name) or "") if name != "files_changed" else " ".join(parse_files(row[name]))
            for name in FIELDS]


def pr_text(row):
    return ". ".join(pr_fields(row))


def overlap(left, right):
    first, second = set(tokens(left)), set(tokens(right))
    return len(first & second) / max(1, len(first | second))


class AlignmentFeatures:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False)
        self.bm25 = None

    def fit(self, rows, tickets):
        corpus = [issue_text(ticket) for ticket in tickets if ticket["split"] == "train"]
        corpus += [pr_text(row) for row in rows if row["split"] == "train"]
        if not corpus:
            raise ValueError("Training text is required for lexical features")
        self.vectorizer.fit(corpus)
        self.bm25 = BM25Okapi([tokens(text) or ["empty"] for text in corpus])
        return self

    def transform(self, rows, tickets, encoder=None, cross_encoder=None):
        if self.bm25 is None:
            raise ValueError("Fit lexical statistics on training data first")
        lookup = {ticket["id"]: ticket for ticket in tickets}
        linked = [lookup[row["jira_id"]] for row in rows]
        jira_texts = [issue_text(ticket) for ticket in linked]
        pr_texts = [pr_text(row) for row in rows]
        jira_vectors = self.vectorizer.transform(jira_texts)
        pr_vectors = self.vectorizer.transform(pr_texts)
        tfidf = np.asarray(jira_vectors.multiply(pr_vectors).sum(axis=1)).ravel()
        scorer = copy.copy(self.bm25)
        document_tokens = [tokens(text) for text in pr_texts]
        scorer.doc_freqs = [Counter(document) for document in document_tokens]
        scorer.doc_len = [len(document) for document in document_tokens]
        scorer.corpus_size = len(rows)
        lexical = []
        for index, (row, ticket) in enumerate(zip(rows, linked)):
            lexical.append([
                tfidf[index], float(scorer.get_batch_scores(tokens(jira_texts[index]), [index])[0]),
                overlap(jira_texts[index], pr_texts[index]),
                float(ticket["module"] in tokens(pr_fields(row)[3])),
                overlap(ticket["module"], row["branch"]),
                float(not row["description"].strip()),
            ])
        values = np.array(lexical, dtype=float)
        names = list(LEXICAL_NAMES)
        if encoder is not None:
            jira_embeddings = encoder.encode(jira_texts)
            for field_index, field in enumerate(FIELDS):
                embeddings = encoder.encode([pr_fields(row)[field_index] for row in rows])
                similarities = np.einsum("ij,ij->i", jira_embeddings, embeddings)
                values = np.column_stack((values, similarities))
                names.append(f"semantic_{field}")
        if cross_encoder is not None:
            scores = np.asarray(cross_encoder.predict(list(zip(jira_texts, pr_texts)))).reshape(-1)
            values = np.column_stack((values, scores))
            names.append("cross_encoder_score")
        if not np.isfinite(values).all():
            raise ValueError("Alignment features must be finite")
        return values, names


class BiEncoder:
    def __init__(self, model_name, batch_size=2, max_length=128, device="cpu", revision=None,
                 purpose="inference"):
        from sentence_transformers import SentenceTransformer
        from .resources import configure_torch, preflight_cuda
        configure_torch()
        if device not in {"cpu", "cuda"}:
            raise ValueError("BiEncoder supports cpu or guarded cuda")
        if device == "cuda" and ((purpose == "training" and batch_size != 1) or max_length > 128):
            raise ValueError("Guarded CUDA BGE requires batch size 1 for training and at most 128 tokens")
        self.gpu_preflight = preflight_cuda(purpose) if device == "cuda" else None
        self.device = device
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.cache = {}

    def encode(self, texts):
        unique = list(dict.fromkeys(text for text in texts if text and text not in self.cache))
        if unique:
            print(f"Encoding {len(unique)} uncached texts on {self.device}; batch size {self.batch_size}", flush=True)
            try:
                encoded = self.model.encode(unique, batch_size=self.batch_size,
                                            normalize_embeddings=True, show_progress_bar=True)
            except Exception as error:
                import torch
                if self.device == "cuda" and isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    raise RuntimeError("CUDA ran out of memory during encoding; the run stopped without retry. Lower the workload or use CPU.") from error
                raise
            self.cache.update(zip(unique, encoded))
        dimension = self.model.get_sentence_embedding_dimension()
        return np.array([self.cache[text] if text else np.zeros(dimension) for text in texts])

    def unload(self):
        import gc
        import torch
        self.model.to("cpu")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load_cross_encoder(model_name, max_length=128, device="cpu", revision=None):
    from sentence_transformers import CrossEncoder
    from .resources import configure_torch, preflight_cuda
    configure_torch()
    if device not in {"cpu", "cuda"}:
        raise ValueError("Cross-encoder supports cpu or guarded cuda")
    if device == "cuda" and max_length > 128:
        raise ValueError("Guarded CUDA cross-encoder supports at most 128 tokens")
    if device == "cuda":
        preflight_cuda("inference")
    return CrossEncoder(model_name, max_length=max_length, device=device, revision=revision)


def train_alignment(features, rows, names, seed=42):
    return train_classifiers(features, rows, "is_aligned", names, seed=seed,
                             invert=True, include_forest=False)


def mine_negative_indices(scores, candidate_groups, positive_groups):
    indices = np.argsort(-np.asarray(scores), kind="stable")
    return [int(index) for index in indices if candidate_groups[index] not in positive_groups]


def finetune(encoder, rows, tickets, output, epochs=1, seed=42):
    import random
    import torch
    from sentence_transformers import losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

    if epochs < 1:
        raise ValueError("Fine-tuning requires at least one epoch")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    lookup = {ticket["id"]: ticket for ticket in tickets}
    positives = {}
    for row in rows:
        if (row["split"] == "train" and row["is_aligned"] == 1
                and lookup[row["jira_id"]]["group_id"] == row["group_id"]):
            positives.setdefault(row["group_id"], row)
    if len(positives) < 2:
        raise ValueError("Fine-tuning requires multiple independent positive groups")
    candidates = [ticket for ticket in tickets if ticket["split"] == "train"]
    candidate_texts = [issue_text(ticket) for ticket in candidates]
    candidate_groups = [ticket["group_id"] for ticket in candidates]
    selected = [row for row in rows if row["split"] == "selection"]
    evaluator = EmbeddingSimilarityEvaluator(
        [issue_text(lookup[row["jira_id"]]) for row in selected],
        [pr_text(row) for row in selected], [float(row["is_aligned"]) for row in selected],
        batch_size=encoder.batch_size, similarity_fn_names=["cosine"], write_csv=False,
    )
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=2e-5)
    loss_function = losses.MultipleNegativesRankingLoss(encoder.model)
    device = encoder.model.device
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    best_score = float("-inf")
    history = []
    for epoch in range(epochs):
        encoder.cache.clear()
        candidate_embeddings = encoder.encode(candidate_texts)
        examples = list(positives.values())
        rng.shuffle(examples)
        anchor_embeddings = encoder.encode([pr_text(row) for row in examples])
        losses_seen = []
        for start in range(0, len(examples), min(encoder.batch_size, 4)):
            batch = examples[start:start + min(encoder.batch_size, 4)]
            groups = {row["group_id"] for row in batch}
            anchors = anchor_embeddings[start:start + len(batch)]
            triples = []
            for row, anchor in zip(batch, anchors):
                possible = mine_negative_indices(candidate_embeddings @ anchor, candidate_groups, groups)
                if not possible:
                    raise ValueError("No training-only hard negatives remain after positive filtering")
                triples.append((pr_text(row), issue_text(lookup[row["jira_id"]]),
                                candidate_texts[possible[0]]))
            encoder.model.train()
            optimizer.zero_grad(set_to_none=True)
            sentence_features = []
            for texts in zip(*triples):
                tokenized = encoder.model.tokenize(list(texts))
                sentence_features.append({key: value.to(device) for key, value in tokenized.items()})
            try:
                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = loss_function(sentence_features, None)
                else:
                    loss = loss_function(sentence_features, None)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError as error:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise RuntimeError("CUDA ran out of memory during fine-tuning; the run stopped without retry. Lower the workload or use CPU.") from error
            losses_seen.append(float(loss.detach().cpu()))
            if len(losses_seen) % 10 == 0 or start + len(batch) >= len(examples):
                print(f"Fine-tuning epoch {epoch + 1}/{epochs}: {start + len(batch)}/{len(examples)} task groups", flush=True)
        encoder.cache.clear()
        result = evaluator(encoder.model)
        score = float(result[evaluator.primary_metric])
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses_seen)), "selection_score": score})
        if score > best_score:
            best_score = score
            encoder.model.save(str(output))
    return history