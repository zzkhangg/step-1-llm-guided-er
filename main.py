from loader import load_data
from attribute_selection.manual import manual_selection
from attribute_selection.heuristics import heuristic_selection
from attribute_selection.llm_guided import llm_guided_selection
import os
import gensim.downloader as api
import numpy as np
from embeddings.embeddings import record_to_vector
from sklearn.preprocessing import normalize
from blocking.lsh import create_random_planes, query_lsh_fast
from constants import *
from matcher.matcher import infer_candidates_pairwise, preview_prompt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils import build_id_maps, build_gt_set
# -----------------------------
# CONFIG
# -----------------------------
SELECTION_METHOD = "heuristics"
# options: manual, heuristic, supervised

ID_A = "id"
ID_B = "id"

# -----------------------------
# Load data
# -----------------------------
df_A = load_data(os.path.join(BASE_PATH, "DBLP.csv"), encoding='latin1')
df_B = load_data(os.path.join(BASE_PATH, "ACM.csv"), encoding='latin1')
df_gt = load_data(os.path.join(BASE_PATH, "gold.csv"), encoding='latin1')

print(df_A.shape)
print(df_B.shape)
# Build ID → position maps
idA_to_pos , idB_to_pos = build_id_maps(df_A, df_B, ID_A, ID_B)
df_A = df_A.drop(columns=[ID_A]).copy()
df_B = df_B.drop(columns=[ID_B]).copy()

# -----------------------------
# Attribute selection
# -----------------------------
if SELECTION_METHOD == "manual":
    df_A, df_B = manual_selection(df_A, df_B)

elif SELECTION_METHOD == "heuristic":
    df_A, df_B = heuristic_selection(df_A, df_B)

elif SELECTION_METHOD == "supervised":
    df_A, df_B = llm_guided_selection(df_A, df_B, df_gt)

print("Selection method:", SELECTION_METHOD)
print("Attributes used:", df_A.columns.tolist())
# check the offending pair directly

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
gt_set = build_gt_set(df_gt, idA_to_pos, idB_to_pos, 'idDBLP', 'idACM')

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

result_df = infer_candidates_pairwise(df_A, df_B, candidate_pairs)

fn_pairs = []
for _, row in result_df.iterrows():
    i, j = int(row['indexA']), int(row['indexB'])
    if (i, j) in gt_set and row['answer'] == 'No':
        fn_pairs.append((i, j))

for i, j in fn_pairs[:10]:
    print("\n--- LLM said No but should be Yes ---")
    print("A:", df_A.iloc[i].to_dict())
    print("B:", df_B.iloc[j].to_dict())

print(f"Total pairs checked by LLM: {len(result_df)}")

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