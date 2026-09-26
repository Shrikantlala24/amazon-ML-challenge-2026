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
    # KEEP unicode letters (Hindi, Tamil, French, etc.)! Do NOT erase with [^a-z0-9]!
    name = re.sub(r'[^\w\s]', ' ', name)
    return ' '.join(name.split())

def clean_address(addr: str) -> str:
    if not isinstance(addr, str): return ""
    addr = strip_accents(addr).lower()
    addr = re.sub(r'\b(street|str)\b', 'st', addr)
    addr = re.sub(r'\b(road)\b', 'rd', addr)
    addr = re.sub(r'\b(avenue|ave\.)\b', 'ave', addr)
    addr = re.sub(r'\b(boulevard|blvd\.)\b', 'blvd', addr)
    addr = re.sub(r'\b(po box|pmb|hn|plot no|house no|flat no|unit|apt|suite)\b[\s\d-]*', ' ', addr)
    addr = re.sub(r'[^\w\s,]', ' ', addr)
    return ' '.join(addr.split())

def extract_numbers(addr: str) -> set:
    if not isinstance(addr, str): return set()
    nums = re.findall(r'\b\d+\b', addr)
    return {str(int(n)) for n in nums if len(n) <= 8}

# Load 1000 ground truth
gt = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for i, row in enumerate(r):
        if i >= 1000: break
        m = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        gt[row[0]] = set(m)

sample_s1_ids = set(gt.keys())
all_target_cands = set()
for m in gt.values(): all_target_cands.update(m)

s1_records = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in sample_s1_ids:
            s1_records[row[0]] = (row[1], row[2])

cand_records = {}
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in all_target_cands:
                cand_records[row[0]] = (row[1], row[2])

# Multi-Key Inverted Index with:
# 1. Name tokens + address
# 2. Name 4-char prefix
# 3. Numbers + address street
# 4. Long numbers (PIN code / phone)
def make_tournament_keys(c_name, c_addr, nums):
    keys = []
    toks = [t for t in c_name.split() if len(t) >= 2]
    addr_toks = [t for t in c_addr.split() if len(t) >= 3]
    
    # 1. First token + address prefix
    if toks:
        addr_p = c_addr[:3] if len(c_addr) >= 3 else "xx"
        keys.append(f"k1:{toks[0]}_{addr_p}")
        if len(toks) >= 2:
            keys.append(f"k2:{'-'.join(sorted(toks[:2]))}")
        for n in nums:
            keys.append(f"k3:{toks[0]}_{n}")
            
    # 2. 4-char compact prefix (catches domain names like maurewilliamscolombier.com)
    compact_name = re.sub(r'\s+', '', c_name)
    if len(compact_name) >= 4:
        keys.append(f"pfx:{compact_name[:4]}")
        
    # 3. Address numbers + street name (catches non-English names & DBA name shifts!)
    if nums and addr_toks:
        for n in nums:
            keys.append(f"addr_num:{addr_toks[0]}_{n}")
            
    # 4. Long numbers (PIN codes, phone numbers >= 5 digits)
    for n in nums:
        if len(n) >= 5:
            keys.append(f"pin:{n}")
            
    return keys

index = defaultdict(list)
for cid, (r_name, r_addr) in cand_records.items():
    cn = clean_name(r_name)
    ca = clean_address(r_addr)
    num = extract_numbers(r_addr)
    for k in make_tournament_keys(cn, ca, num):
        index[k].append(cid)

hits = 0
total_targets = 0
for s1_id, targets in gt.items():
    if not targets: continue
    s1_rname, s1_raddr = s1_records[s1_id]
    cn = clean_name(s1_rname)
    ca = clean_address(s1_raddr)
    num = extract_numbers(s1_raddr)
    
    keys = make_tournament_keys(cn, ca, num)
    counts = Counter()
    for k in keys:
        for cid in index.get(k, []):
            counts[cid] += 1
    found = {cid for cid, _ in counts.most_common(6)}
    
    for t in targets:
        if t in cand_records:
            total_targets += 1
            if t in found:
                hits += 1

print(f"\n==========================================")
print(f"🚀 NEW TOURNAMENT RECALL: {hits} / {total_targets} = {hits / total_targets:.2%}")
print(f"==========================================\n")
