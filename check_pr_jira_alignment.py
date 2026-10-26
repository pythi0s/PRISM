import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os

# === Constants ===
JIRA_FILE = "jira_tickets.csv"
PR_FILE = "pull_requests.csv"
OUTPUT_FILE = "alignment_scores.csv"
THRESHOLD = 0.7
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load Input Files ===
if not all(os.path.exists(f) for f in [JIRA_FILE, PR_FILE]):
    raise FileNotFoundError("One or both input files are missing.")

jira_df = pd.read_csv(JIRA_FILE)
pr_df = pd.read_csv(PR_FILE)

if jira_df.shape[0] != 500 or pr_df.shape[0] != 500:
    raise ValueError(f"Expected 500 records each. Got: JIRA={len(jira_df)}, PR={len(pr_df)}")

# === Merge on JIRA ID ===
df = pd.merge(jira_df, pr_df, left_on="id", right_on="jira_id", suffixes=("_jira", "_pr"))
if df.shape[0] != 500:
    raise ValueError(f"Merged records count mismatch. Expected 500, got {df.shape[0]}")

# === Load SBERT Model (GPU if available) ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SBERT model on device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

# === Prepare Texts ===
jira_texts = df["description_jira"].astype(str).tolist()
pr_texts = df["description_pr"].astype(str).tolist()

# === Encode Descriptions ===
print("Encoding descriptions...")
jira_embeddings = model.encode(jira_texts, convert_to_tensor=True, show_progress_bar=True)
pr_embeddings = model.encode(pr_texts, convert_to_tensor=True, show_progress_bar=True)

# === Compute Diagonal Cosine Similarities ===
print("Computing cosine similarities...")
similarity_scores = util.cos_sim(jira_embeddings, pr_embeddings).diagonal().cpu().numpy()

# === Build Output DataFrame ===
output_df = df[[
    "id", "pr_id", "description_jira", "description_pr",
    "lines_added", "lines_deleted", "files_changed", "comment_count"
]].copy()

output_df.rename(columns={
    "id": "jira_id",
    "description_jira": "jira_description",
    "description_pr": "pr_description"
}, inplace=True)

output_df["similarity_score"] = similarity_scores

# === Save to CSV ===
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved results to {OUTPUT_FILE}")

# === Summary ===
low_scores = output_df[output_df["similarity_score"] < THRESHOLD]
print("\n--- Alignment Summary ---")
print(f"Total Pairs: {len(output_df)}")
print(f"Average Similarity Score: {similarity_scores.mean():.4f}")
print(f"Pairs < {THRESHOLD}: {len(low_scores)}")

if not low_scores.empty:
    print("\nTop 5 Low-Scoring Pairs:")
    print(low_scores[["jira_id", "pr_id", "similarity_score"]].head().to_string(index=False))
else:
    print("All pairs have acceptable alignment scores.")

# === Plot Similarity Score Distribution ===
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(output_df["similarity_score"], bins=30, color='skyblue', edgecolor='black')
plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold = {THRESHOLD}')
plt.title("Distribution of Similarity Scores")
plt.xlabel("Similarity Score")
plt.ylabel("Number of JIRA-PR Pairs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
