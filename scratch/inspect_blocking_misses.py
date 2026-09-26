import os
import csv
import re
import unicodedata
from collections import defaultdict, Counter

TRAIN_DIR = os.path.abspath(r"dataset_folder/student_resource/dataset/train")

def strip_accents(text: str) -> str:
    if not isinstance(text, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_name(name: str) -> str:
    if not isinstance(name, str): return ""
    name = strip_accents(name).lower()
    name = re.sub(r'(@\w+|https?://\S+|www\.\S+|\.com|\.org|\.in|\.fr)', ' ', name)
    name = re.sub(r'\b(limited liability company|llc|incorporated|inc|corporation|corp|private limited|pvt ltd|pvt\. ltd\.|limited|ltd|llp|societe a responsabilite limitee|sarl|sasu|sas|eurl|sci|association|assoc|cie|et fils|& fils)\b', ' ', name)
    name = re.sub(r'\b(d/?b/?a|trading as|t/a)\b.*', ' ', name)
    name = re.sub(r'\([\w\s:]+\)|\[[\w\s:]+\]', ' ', name)
    name = re.sub(r'(?<=[a-z])1\b', 'l', name)
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    return ' '.join(name.split())

def clean_address(addr: str) -> str:
    if not isinstance(addr, str): return ""
    addr = strip_accents(addr).lower()
    addr = re.sub(r'\b(street|str)\b', 'st', addr)
    addr = re.sub(r'\b(road)\b', 'rd', addr)
    addr = re.sub(r'\b(avenue|ave\.)\b', 'ave', addr)
    addr = re.sub(r'\b(boulevard|blvd\.)\b', 'blvd', addr)
    addr = re.sub(r'\b(po box|pmb|hn|plot no|house no|flat no|unit|apt|suite)\b[\s\d-]*', ' ', addr)
    addr = re.sub(r'[^a-z0-9\s,]', ' ', addr)
    return ' '.join(addr.split())

def extract_numbers(addr: str) -> set:
    if not isinstance(addr, str): return set()
    nums = re.findall(r'\b\d+\b', addr)
    return {str(int(n)) for n in nums if len(n) <= 6}

# Load 500 ground truth
gt = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for i, row in enumerate(r):
        if i >= 500: break
        m = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        gt[row[0]] = set(m)

sample_s1_ids = set(gt.keys())
all_target_cands = set()
for m in gt.values(): all_target_cands.update(m)

# Load S1 records
s1_records = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in sample_s1_ids:
            s1_records[row[0]] = (row[1], row[2])

# Load candidate records
cand_records = {}
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in all_target_cands:
                cand_records[row[0]] = (row[1], row[2])

# Check why they are missed by MasterInvertedIndex
def _make_keys(c_name, c_addr, nums):
    keys = []
    toks = [t for t in c_name.split() if len(t) >= 2]
    if not toks: return keys
    addr_p = c_addr[:3] if len(c_addr) >= 3 else "xx"
    keys.append(f"k1:{toks[0]}_{addr_p}")
    for n in nums: keys.append(f"k2:{toks[0]}_{n}")
    if len(toks) >= 2: keys.append(f"k3:{'-'.join(sorted(toks[:2]))}")
    if len(toks) >= 2 and nums:
        for n in nums: keys.append(f"k4:{toks[1]}_{n}")
    if len(c_name) <= 22: keys.append(f"k5:{c_name}")
    return keys

index = defaultdict(list)
for cid, (r_name, r_addr) in cand_records.items():
    cn = clean_name(r_name)
    ca = clean_address(r_addr)
    num = extract_numbers(r_addr)
    for k in _make_keys(cn, ca, num):
        index[k].append(cid)

missed = []
for s1_id, targets in gt.items():
    if not targets: continue
    s1_rname, s1_raddr = s1_records[s1_id]
    cn = clean_name(s1_rname)
    ca = clean_address(s1_raddr)
    num = extract_numbers(s1_raddr)
    
    keys = _make_keys(cn, ca, num)
    counts = Counter()
    for k in keys:
        for cid in index.get(k, []):
            counts[cid] += 1
    found = {cid for cid, _ in counts.most_common(5)}
    
    for t in targets:
        if t not in found and t in cand_records:
            t_rname, t_raddr = cand_records[t]
            missed.append((s1_rname, s1_raddr, t_rname, t_raddr))

print(f"Total missed targets analyzed: {len(missed)}")
print("--- 10 Examples of Missed Matches ---")
for i, (s1_n, s1_a, c_n, c_a) in enumerate(missed[:10]):
    print(f"\nExample #{i+1}:")
    print(f"  S1   : '{s1_n}' | '{s1_a}'")
    print(f"  Cand : '{c_n}' | '{c_a}'")
    print(f"  Clean S1  : '{clean_name(s1_n)}'")
    print(f"  Clean Cand: '{clean_name(c_n)}'")
