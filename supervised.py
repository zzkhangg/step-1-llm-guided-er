from loader import load_data
from attribute_selection import manual_selection, llm_guided_selection, heuristic_selection, supervised_selection
import os
import gensim.downloader as api
import numpy as np
from embeddings.embeddings import record_to_vector
from sklearn.preprocessing import normalize
from blocking.lsh import create_random_planes, query_lsh_fast
from constants import *
from matcher.matcher import infer_candidates_pairwise
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import build_id_maps, build_gt_set
# -----------------------------
# CONFIG
# -----------------------------
SELECTION_METHOD = "supervised"
# options: manual, heuristic, supervised, llm_guided

ID_A = "id"
ID_B = "id"

# -----------------------------
# Load data
# -----------------------------
df_A = load_data(os.path.join(BASE_PATH, "amazon.csv"))
df_B = load_data(os.path.join(BASE_PATH, "walmart.csv"))
df_gt = load_data(os.path.join(BASE_PATH, "gold.tsv"))

print(df_A.shape)
print(df_B.shape)
# Build ID → position maps
idA_to_pos , idB_to_pos = build_id_maps(df_A, df_B, ID_A, ID_B)
df_A = df_A.drop(columns=[ID_A]).copy()
df_B = df_B.drop(columns=[ID_B]).copy()

# -----------------------------
# Embeddings
# -----------------------------
glove = api.load("glove-wiki-gigaword-300")
np.random.seed(42)

tableA_vectors = np.vstack([
    record_to_vector(row, glove)
    for _, row in df_A.iterrows()
])

tableB_vectors = np.vstack([
    record_to_vector(row, glove)
    for _, row in df_B.iterrows()
])

tableA_vectors = normalize(tableA_vectors)
tableB_vectors = normalize(tableB_vectors)

print("TableA vectors shape:", tableA_vectors.shape)
print("TableB vectors shape:", tableB_vectors.shape)

# -----------------------------
# Attribute selection
# -----------------------------
if SELECTION_METHOD == "manual":
    df_A, df_B = manual_selection(df_A, df_B)

elif SELECTION_METHOD == "heuristic":
    df_A, df_B = heuristic_selection(df_A, df_B)

elif SELECTION_METHOD == "supervised":
    df_A, df_B, ranked = supervised_selection(
        df_A=df_A,
        df_B=df_B,
        df_train=df_gt,        # your gold labels
        id_col_A="id1",
        id_col_B="id2",
        idA_to_pos=idA_to_pos,
        idB_to_pos=idB_to_pos,
        tableA_vectors=tableA_vectors,
        tableB_vectors=tableB_vectors,
        planes_list=create_random_planes(
            num_tables=15,
            num_planes=6,
            dim=tableA_vectors.shape[1]
        ),
        blocking_top_k=5,
        neg_ratio=1.0,
        threshold=0.7,
        save_model=False
    )
elif SELECTION_METHOD == "llm_guided":
    df_A, df_B = llm_guided_selection(df_A, df_B, df_gt)

print("Selection method:", SELECTION_METHOD)
print("Attributes used:", df_A.columns.tolist())
# check the offending pair directly

# recompute embeddings AFTER selection
tableA_vectors = np.vstack([
    record_to_vector(row, glove)
    for _, row in df_A.iterrows()
])

tableB_vectors = np.vstack([
    record_to_vector(row, glove)
    for _, row in df_B.iterrows()
])

tableA_vectors = normalize(tableA_vectors)
tableB_vectors = normalize(tableB_vectors)

dim = tableA_vectors.shape[1]

# create hash functions
planes_list = create_random_planes(num_tables=15, num_planes=6, dim=dim)

candidate_pairs  = query_lsh_fast(tableA_vectors, tableB_vectors, planes_list, num_flips=1, top_k=5)

print("Number of candidate pairs after top-k:", len(candidate_pairs))

n_A = tableA_vectors.shape[0]
n_B = tableB_vectors.shape[0]

total_pairs = n_A * n_B
candidate_pairs_count = len(candidate_pairs)

reduction_percentage = (total_pairs - candidate_pairs_count) / total_pairs * 100

print(f"Reduction percentage: {reduction_percentage:.2f}%")

# Convert ground truth to positional indices
gt_set = build_gt_set(df_gt, idA_to_pos, idB_to_pos, 'id1', 'id2')

cand_set = set(candidate_pairs)
found = sum(1 for pair in gt_set if pair in cand_set)
pc = found / len(gt_set)
print(f"True matches:      {len(gt_set)}")
print(f"Found in blocking: {found}")
print(f"Pair Completeness: {pc:.4f}")
print(f"Candidates:        {len(candidate_pairs)}")

# ---------------------------------------
# Call LLM for candidate pairs
# ---------------------------------------
# You can adjust max_workers based on API rate limits
print(df_A.columns.tolist())
result_df = infer_candidates_pairwise(df_A, df_B, candidate_pairs)

# Inference 
# ---------------------------------------
# Align result_df with ground truth
# ---------------------------------------

# Map predictions to binary labels
y_pred = []
y_true = []

for _, row in result_df.iterrows():
    i, j = int(row['indexA']), int(row['indexB'])
    y_pred.append(1 if row['answer'] == 'Yes' else 0)
    y_true.append(1 if (i, j) in gt_set else 0)

# ---------------------------------------
# Metrics
# ---------------------------------------
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

total_prompt_tokens = result_df["prompt_tokens"].sum()
total_completion_tokens = result_df["completion_tokens"].sum()
total_tokens = result_df["total_tokens"].sum()

print("\n--- Token Usage ---")
print(f"Prompt tokens     : {total_prompt_tokens}")
print(f"Completion tokens : {total_completion_tokens}")
print(f"Total tokens      : {total_tokens}")
print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ---------------------------------------
# How many true matches were in candidates vs found by LLM
# ---------------------------------------
cand_set  = set(zip(result_df['indexA'], result_df['indexB']))
in_cands  = sum(1 for pair in gt_set if pair in cand_set)
found_llm = tp

print(f"\nTrue matches total         : {len(gt_set)}")
print(f"True matches in candidates : {in_cands}  (blocking recall)")
print(f"True matches found by LLM  : {found_llm}  (end-to-end recall)")