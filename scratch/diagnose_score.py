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

# Load ground truth for US sample
gt = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for i, row in enumerate(r):
        if i >= 5000: break
        m = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        gt[row[0]] = set(m)

sample_s1_ids = set(gt.keys())
print(f"Sampled {len(sample_s1_ids)} S1 ground truth entities.")
singletons = sum(1 for v in gt.values() if len(v) == 0)
has_matches = sum(1 for v in gt.values() if len(v) > 0)
total_matches = sum(len(v) for v in gt.values())
print(f"Singletons (no match): {singletons} ({singletons/len(gt):.1%})")
print(f"Has matches: {has_matches} ({has_matches/len(gt):.1%}) with {total_matches} total linked entities.")
