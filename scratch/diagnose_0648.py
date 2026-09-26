import os
import csv
import re
import unicodedata
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz

TRAIN_DIR = os.path.abspath(r"dataset_folder/student_resource/dataset/train")

def strip_accents(text: str) -> str:
    if not isinstance(text, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

TAG_PATTERNS = {
    'llc': r'\b(limited liability company|llc)\b',
    'inc': r'\b(incorporated|inc|corporation|corp)\b',
    'ltd': r'\b(private limited|pvt ltd|pvt\. ltd\.|limited|ltd|llp)\b',
    'sarl': r'\b(societe a responsabilite limitee|sarl|sasu|sas|eurl|sci)\b',
    'assoc': r'\b(association|assoc|cie|et fils|& fils)\b'
}
ALL_TAGS_REGEX = re.compile('|'.join(TAG_PATTERNS.values()), flags=re.IGNORECASE)

def extract_legal_tag(name: str) -> str:
    if not isinstance(name, str): return "none"
    name_lower = strip_accents(name).lower()
    for tag_name, pat in TAG_PATTERNS.items():
        if re.search(pat, name_lower): return tag_name
    return "none"

def clean_name(name: str) -> str:
    if not isinstance(name, str): return ""
    name = strip_accents(name).lower()
    name = re.sub(r'(@\w+|https?://\S+|www\.\S+|\.com|\.org|\.in|\.fr)', ' ', name)
    name = ALL_TAGS_REGEX.sub(' ', name)
    name = re.sub(r'\b(d/?b/?a|trading as|t/a)\b.*', ' ', name)
    name = re.sub(r'\([\w\s:]+\)|\[[\w\s:]+\]', ' ', name)
    name = re.sub(r'(?<=[a-z])1\b', 'l', name)
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    return ' '.join(name.split())

ROAD_MAP = {
    r'\b(street|str)\b': 'st', r'\b(road)\b': 'rd', r'\b(avenue|ave\.)\b': 'ave',
    r'\b(boulevard|blvd\.)\b': 'blvd', r'\b(drive|dr\.)\b': 'dr', r'\b(lane|ln\.)\b': 'ln',
    r'\b(highway|hwy\.)\b': 'hwy', r'\b(rue|r\.)\b': 'rue', r'\b(allee|all\.)\b': 'allee'
}

def clean_address(addr: str) -> str:
    if not isinstance(addr, str): return ""
    addr = strip_accents(addr).lower()
    for pat, rep in ROAD_MAP.items():
        addr = re.sub(pat, rep, addr)
    addr = re.sub(r'\b(po box|pmb|hn|plot no|house no|flat no|unit|apt|suite)\b[\s\d-]*', ' ', addr)
    addr = re.sub(r'[^a-z0-9\s,]', ' ', addr)
    return ' '.join(addr.split())

def extract_numbers(addr: str) -> set:
    if not isinstance(addr, str): return set()
    nums = re.findall(r'\b\d+\b', addr)
    return {str(int(n)) for n in nums if len(n) <= 6}

def get_char_3grams(text):
    text = f" {text} "
    if len(text) < 3: return {text}
    return {text[i:i+3] for i in range(len(text)-2)}

def compute_clues(s1_raw_name, s1_clean_name, s1_tag, s1_raw_addr, s1_clean_addr, s1_nums,
                  cand_raw_name, cand_clean_name, cand_tag, cand_raw_addr, cand_clean_addr, cand_nums,
                  cand_id):
    feats = {}
    feats['clean_name_sort'] = fuzz.token_sort_ratio(s1_clean_name, cand_clean_name) / 100.0
    feats['clean_name_lev']  = fuzz.ratio(s1_clean_name, cand_clean_name) / 100.0
    feats['clean_name_set']  = fuzz.token_set_ratio(s1_clean_name, cand_clean_name) / 100.0
    s1_ngs = get_char_3grams(s1_clean_name)
    cand_ngs = get_char_3grams(cand_clean_name)
    inter = len(s1_ngs & cand_ngs)
    union = len(s1_ngs | cand_ngs)
    feats['ngram_jaccard']   = inter / union if union > 0 else 0.0
    s1_toks = s1_clean_name.split()
    cand_toks = cand_clean_name.split()
    feats['first_word_match'] = 1.0 if (s1_toks and cand_toks and s1_toks[0] == cand_toks[0]) else 0.0
    if s1_tag == cand_tag and s1_tag != 'none': feats['tag_score'] = 1.0
    elif s1_tag == 'none' or cand_tag == 'none': feats['tag_score'] = 0.5
    else: feats['tag_score'] = 0.0
    feats['has_empty_addr']  = 1.0 if (not s1_clean_addr or not cand_clean_addr) else 0.0
    if feats['has_empty_addr']:
        feats['addr_token_sort'] = 0.0
        feats['number_jaccard']  = 0.0
        feats['number_exact_match'] = 0.0
    else:
        feats['addr_token_sort'] = fuzz.token_sort_ratio(s1_clean_addr, cand_clean_addr) / 100.0
        num_union = s1_nums | cand_nums
        feats['number_jaccard'] = len(s1_nums & cand_nums) / len(num_union) if num_union else 0.5
        feats['number_exact_match'] = 1.0 if (s1_nums and cand_nums and s1_nums == cand_nums) else 0.0
    feats['is_source2'] = 1.0 if cand_id.startswith('S2-') else 0.0
    feats['is_source3'] = 1.0 if cand_id.startswith('S3-') else 0.0
    return feats

def calc_f05(true_set, pred_set):
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0, "true_empty"
    if len(pred_set) == 0 and len(true_set) > 0:
        return 0.0, "missed_all"
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0, "false_merge"
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0, "zero_tp"
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f05 = 1.25 * p * r / (0.25 * p + r)
    return f05, "matched"

print("1. Loading ground truth for 3,000 validation entities...")
gt = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for i, row in enumerate(r):
        if i >= 3000: break
        m = row[1].split(",") if len(row) > 1 and row[1].strip() else []
        gt[row[0]] = set(m)

val_s1_ids = set(gt.keys())
all_target_cands = set()
for m in gt.values():
    all_target_cands.update(m)

print(f"Loaded {len(gt)} entities. Total true target matches: {len(all_target_cands)}")

print("2. Building index for candidate records...")
class MasterInvertedIndex:
    def __init__(self, max_candidates=5):
        self.index = defaultdict(list)
        self.records = {}
        self.max_candidates = max_candidates
        
    def _make_keys(self, c_name, c_addr, nums):
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

    def add_record(self, eid, raw_name, raw_addr):
        c_name = clean_name(raw_name)
        c_addr = clean_address(raw_addr)
        nums = extract_numbers(raw_addr)
        tag = extract_legal_tag(raw_name)
        self.records[eid] = (raw_name, c_name, tag, raw_addr, c_addr, nums)
        for k in self._make_keys(c_name, c_addr, nums):
            self.index[k].append(eid)

    def get_candidates(self, c_name, c_addr, nums):
        keys = self._make_keys(c_name, c_addr, nums)
        counts = Counter()
        for k in keys:
            for eid in self.index.get(k, []):
                counts[eid] += 1
        if not counts: return []
        return [eid for eid, _ in counts.most_common(self.max_candidates)]

engine = MasterInvertedIndex(max_candidates=5)
# Load true target candidates + 100k noise
loaded = 0
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(TRAIN_DIR, fname), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in all_target_cands or loaded < 100000:
                engine.add_record(row[0], row[1], row[2])
                loaded += 1
            if loaded >= 120000: break

print(f"Indexed {len(engine.records):,} records.")

# Check Blocking Recall: how many true target matches are in the candidate basket?
blocking_hits = 0
total_targets = 0
val_s1 = {}
with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in val_s1_ids:
            val_s1[row[0]] = (row[1], row[2])

for s1_id, true_set in gt.items():
    if not true_set: continue
    raw_name, raw_addr = val_s1[s1_id]
    c_name, c_addr = clean_name(raw_name), clean_address(raw_addr)
    nums = extract_numbers(raw_addr)
    cands = engine.get_candidates(c_name, c_addr, nums)
    c_set = set(cands)
    for t in true_set:
        total_targets += 1
        if t in c_set:
            blocking_hits += 1

print(f"\n==========================================")
print(f"📊 BLOCKING RECALL: {blocking_hits} / {total_targets} = {blocking_hits / total_targets:.2%}")
print(f"==========================================\n")
