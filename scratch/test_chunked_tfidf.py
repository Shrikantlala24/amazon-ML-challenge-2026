import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import psutil
import os

# Simulate 500,000 candidate strings
print("Generating dummy candidate texts...")
cand_texts = [f"business store {i} main street suite {i%100}" for i in range(200000)]
query_texts = [f"business store {i} main st" for i in range(5000)]

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, max_features=50000)
t0 = time.time()
cand_matrix = vec.fit_transform(cand_texts)
print(f"Cand matrix built: {cand_matrix.shape} in {time.time()-t0:.2f}s")

# Test chunked retrieval with batch_size = 1000
query_matrix = vec.transform(query_texts)

CHUNK_SIZE = 1000
all_matches = []
t1 = time.time()
for start in range(0, query_matrix.shape[0], CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, query_matrix.shape[0])
    q_chunk = query_matrix[start:end]
    # Dot product
    sim = q_chunk.dot(cand_matrix.T)
    # Extract top 5 for each row in chunk
    for r in range(sim.shape[0]):
        row = sim.getrow(r)
        if row.nnz > 0:
            top_idx = row.indices[np.argsort(-row.data)[:5]]
            all_matches.append(top_idx)
        else:
            all_matches.append([])
    del sim

print(f"Queried {len(query_texts)} items in {time.time()-t1:.2f}s.")
proc = psutil.Process()
print(f"RAM used: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
