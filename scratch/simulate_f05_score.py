import csv
import re
import unicodedata
from collections import defaultdict, Counter
import sys

def normalize_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('ASCII').lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'\b(d\.b\.a|dba|d/b/a)\b.*', ' ', text)
    text = re.sub(r'\b(corp|corporation|inc|incorporated|llc|l\.l\.c|ltd|limited|co|company|services|partners|private|pvt)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def extract_addr_features(addr):
    if not addr: return set(), set()
    norm = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('ASCII').lower()
    norm = re.sub(r'\b(street|st|saint)\b', 'st', norm)
    norm = re.sub(r'\b(road|rd)\b', 'rd', norm)
    norm = re.sub(r'\b(avenue|ave)\b', 'ave', norm)
    norm = re.sub(r'\b(drive|dr)\b', 'dr', norm)
    tokens = re.findall(r'[a-z0-9]+', norm)
    numbers = set()
    for t in tokens:
        if t.isdigit():
            numbers.add(t.lstrip('0') or '0')
    stopwords = {'north','south','east','west','rd','st','ave','dr','suite','floor','near','behind','door','block','sector','hno','plot','flat','shop','null'}
    words = {t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in stopwords}
    return numbers, words

def token_similarity(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / max(len(t1), len(t2))

def is_match(s1_name, s1_addr, c_name, c_addr):
    # Normalized strings
    n1 = normalize_text(s1_name)
    n2 = normalize_text(c_name)
    nums1, words1 = extract_addr_features(s1_addr)
    nums2, words2 = extract_addr_features(c_addr)
    
    # 1. Exact or near-exact clean name match
    if n1 and n2:
        if n1 == n2:
            return True
        sim = token_similarity(n1, n2)
        if sim >= 0.75:
            return True
        # Check domain / concatenated match
        concat1 = n1.replace(' ', '')
        concat2 = n2.replace(' ', '')
        if concat1 in concat2 or concat2 in concat1:
            if len(min(concat1, concat2)) >= 5:
                return True
                
    # 2. Strong address match (same house number + matching street tokens)
    common_nums = nums1 & nums2
    common_words = words1 & words2
    if common_nums and common_words:
        return True
    if len(common_words) >= 3:
        return True
    # PIN / Phone match (>= 5 digits)
    for num in common_nums:
        if len(num) >= 5:
            return True
            
    return False

print("Simulating F0.5 on 5,000 S1 records...")
gt = {}
with open('dataset_folder/student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if len(row) >= 2 and row[1].strip():
            gt[row[0]] = set(row[1].split(','))
        else:
            gt[row[0]] = set()
        if len(gt) >= 5000:
            break

needed_s1 = set(gt.keys())
needed_targets = {t for s in gt.values() for t in s}

s1_records = {}
with open('dataset_folder/student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if row[0] in needed_s1:
            s1_records[row[0]] = (row[1], row[2], row[3])

cand_records = {}
for s_file in ['train_source2.tsv', 'train_source3.tsv']:
    with open('dataset_folder/student_resource/dataset/train/' + s_file, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in needed_targets:
                cand_records[row[0]] = (row[1], row[2], row[3])

# Calculate Macro F0.5
f05_scores = []
for s1_id, true_matches in gt.items():
    s1_name, s1_addr, _ = s1_records[s1_id]
    
    # In an ideal candidate pool (where candidates are present):
    pred_matches = set()
    for cid in true_matches:
        if cid in cand_records:
            c_name, c_addr, _ = cand_records[cid]
            if is_match(s1_name, s1_addr, c_name, c_addr):
                pred_matches.add(cid)
                
    # F0.5 calculation for this record
    tp = len(pred_matches & true_matches)
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)
    
    if tp == 0 and fp == 0 and fn == 0:
        f05 = 1.0 # both empty
    elif tp == 0:
        f05 = 0.0
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        beta_sq = 0.5 ** 2 # 0.25
        f05 = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec)
        
    f05_scores.append(f05)

macro_f05 = sum(f05_scores) / len(f05_scores)
print(f"Simulated Macro F0.5 Score: {macro_f05:.4f} ({macro_f05*100:.2f}%)")
