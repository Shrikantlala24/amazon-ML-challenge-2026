import time
import psutil
from collections import defaultdict, Counter

print("Testing custom memory-light Character 3-Gram Inverted Index...")

# Dummy 200,000 candidate strings
cands = [(f"cand_{i}", f"john doe pharmacy store {i}", f"123 main street unit {i%50}") for i in range(200000)]

t0 = time.time()
index = defaultdict(list)
records = {}

def get_char_3grams(text):
    text = f" {text.lower()} "
    return {text[i:i+3] for i in range(len(text)-2)}

for cid, name, addr in cands:
    records[cid] = (name, addr)
    ngs = get_char_3grams(name)
    for ng in ngs:
        index[ng].append(cid)

t1 = time.time()
proc = psutil.Process()
ram_mb = proc.memory_info().rss / 1024 / 1024
print(f"Indexed 200,000 records in {t1-t0:.2f}s.")
print(f"Unique 3-grams: {len(index):,}")
print(f"Total RAM used: {ram_mb:.1f} MB")
