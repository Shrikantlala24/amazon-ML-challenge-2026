import os
import csv
import time
from collections import defaultdict, Counter
import psutil

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_lean_blocking import UltraFastLeanBlocking, clean_name, clean_address, extract_numbers

TRAIN_DIR = os.path.abspath(r"dataset_folder/student_resource/dataset/train")

# Load 3000 ground truth
gt = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for i, row in enumerate(r):
        if i >= 3000: break
        m = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        gt[row[0]] = set(m)

sample_s1_ids = set(gt.keys())
all_target_cands = set()
for s in gt.values():
    all_target_cands.update(s)

print(f"Targeting {len(all_target_cands)} true matches across {len(sample_s1_ids)} S1 records.")

# Index 100,000 candidates from train_source2 and train_source3
engine = UltraFastLeanBlocking(max_candidates=8)

loaded_cands = 0
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            # load true candidates + background noise up to 100k
            if row[0] in all_target_cands or loaded_cands < 100000:
                engine.add_record(row[0], row[1], row[2])
                loaded_cands += 1
            if loaded_cands >= 120000:
                break

print(f"Engine indexed {len(engine.records):,} candidate records.")

# Test recall on S1
s1_records = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in sample_s1_ids:
            s1_records[row[0]] = (row[1], row[2])

hits = 0
total_targets = 0
for s1_id, true_set in gt.items():
    if not true_set:
        continue
    raw_name, raw_addr = s1_records[s1_id]
    c_name = clean_name(raw_name)
    c_addr = clean_address(raw_addr)
    nums = extract_numbers(raw_addr)
    
    cands = engine.get_candidates(c_name, c_addr, nums)
    c_set = set(cands)
    for t in true_set:
        total_targets += 1
        if t in c_set:
            hits += 1

print(f"Recall: {hits} / {total_targets} = {hits / total_targets:.2%}")
process = psutil.Process()
print(f"RAM used: {process.memory_info().rss / 1024 / 1024:.1f} MB")
