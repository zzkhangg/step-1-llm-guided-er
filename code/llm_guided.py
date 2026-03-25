
import os
import gensim.downloader as api
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import normalize

from .lsh import create_random_planes, query_lsh_fast
from .utils import generate_negative_pairs, load_negative_pairs, build_labeled_pairs, build_id_maps, build_gt_set
from .constants import *
from .loader import load_data
from .attribute_selection import manual_selection, llm_guided_selection, heuristic_selection, supervised_selection
from .matcher import infer_candidates_pairwise
from .embeddings import record_to_vector



# -----------------------------
# CONFIG
# -----------------------------
SELECTION_METHOD = "llm_guided"
# options: manual, heuristic, supervised, llm_guided

ID_A_COL = "id"
ID_B_COL = "id"
GT_ID_A_COL = 'ltable_id'
GT_ID_B_COL = 'rtable_id'

# -----------------------------
# Load data
# -----------------------------
df_A = load_data(os.path.join(BASE_PATH, "tableA.csv"))
df_B = load_data(os.path.join(BASE_PATH, "tableB.csv"))
df_gt = load_data(os.path.join(BASE_PATH, "gold.csv"))

print(df_A.shape)
print(df_B.shape)
# Build ID → position maps
idA_to_pos , idB_to_pos = build_id_maps(df_A, df_B, ID_A_COL, ID_B_COL)
df_A = df_A.drop(columns=[ID_A_COL]).copy()
df_B = df_B.drop(columns=[ID_B_COL]).copy()
# Build ground truth set from positive pairs
gt_set = build_gt_set(df_gt, idA_to_pos, idB_to_pos, GT_ID_A_COL, GT_ID_B_COL)

# -----------------------------
# Embeddings
# -----------------------------
glove = api.load("glove-wiki-gigaword-300")
seed = np.random.seed(42) ### Reproducible

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

NEGATIVES_PATH = os.path.join(BASE_PATH, "negatives.csv")

# ── run blocking first ──
planes_list     = create_random_planes(num_tables=15, num_planes=6, dim=tableA_vectors.shape[1], seed=seed)
candidate_pairs = query_lsh_fast(tableA_vectors, tableB_vectors, planes_list,
                                 num_flips=1, top_k=3)


# ── generate and save negatives (only once) ──
if not os.path.exists(NEGATIVES_PATH):
    df_negatives = generate_negative_pairs(candidate_pairs, gt_set,
                                           save_path=NEGATIVES_PATH)
else:
    df_negatives = load_negative_pairs(NEGATIVES_PATH)   # reuse on subsequent runs

# ── build labeled pairs for attribute selection ──
labeled_pairs = build_labeled_pairs(
    gt_set, df_negatives,
    n_pos=50,    # sample 50 positives
    n_neg=50,    # sample 50 hard negatives
    seed=42
)

# ── pass to LLM-guided selection ──
df_A, df_B, ranked = llm_guided_selection(
    df_A, df_B, labeled_pairs=labeled_pairs,
    n_pos=10, n_neg=10,
    threshold=0.3
)

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
gt_set = build_gt_set(df_gt, idA_to_pos, idB_to_pos, GT_ID_A_COL, GT_ID_B_COL)

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

