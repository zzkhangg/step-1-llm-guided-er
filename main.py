from loader import load_data
from manual_inspection.filter import filter_attributes
import os
import gensim.downloader as api
import numpy as np
from embeddings import record_to_vector
from sklearn.preprocessing import normalize
from blocking import create_random_planes, query_lsh_fast
from constants import *
from matcher import infer_candidates_batch, infer_pairs_batch

# Load pre-trained GloVe embeddings
glove = api.load("glove-wiki-gigaword-300")  # 300-dimensional embeddings
np.random.seed(42) ## for reproducibility

df_A = load_data(os.path.join(BASE_PATH, "tableA.csv"))
df_B = load_data(os.path.join(BASE_PATH, "tableB.csv"))
df_gt = load_data(os.path.join(BASE_PATH, "gold.csv"))

print(df_A.shape)
print(df_B.shape)
print(df_A.index.tolist()[:5])   # should be [0, 1, 2, 3, 4]
print(df_gt['ltable_id'].head())  # should also start from 0

df_A = filter_attributes(df_A, MARKERS_ATTRIBUTES)
df_B = filter_attributes(df_B, MARKERS_ATTRIBUTES)

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

# assuming you have ground truth as list of (idA, idB) tuples
gt = list(zip(df_gt['ltable_id'], df_gt['rtable_id']))
cand_set = set(candidate_pairs)
found = sum(1 for pair in gt if pair in cand_set)
pc = found / len(gt)
print(f"True matches:      {len(gt)}")
print(f"Found in blocking: {found}")
print(f"Pair Completeness: {pc:.4f}")
print(f"Candidates:        {len(candidate_pairs)}")

missed = [(a, b) for a, b in gt if (a, b) not in cand_set]
for idx_a, idx_b in missed[:10]:
    print("A:", df_A.iloc[idx_a][MARKERS_ATTRIBUTES].tolist())
    print("B:", df_B.iloc[idx_b][MARKERS_ATTRIBUTES].tolist())
    print()

# # ---------------------------------------
# # Call LLM for candidate pairs
# # ---------------------------------------
# # You can adjust max_workers based on API rate limits
# max_workers = 16  

# result_df = infer_candidates_concurrent(df_A, df_B, candidate_pairs, max_workers=max_workers)

# print(result_df.head())
# print(f"Total pairs checked by LLM: {len(result_df)}")