import os
import csv
import re
import unicodedata
from collections import defaultdict, Counter

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('ASCII').lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'\b(d\.b\.a|dba|d/b/a)\b.*', ' ', text) # often DBA precedes original name
    text = re.sub(r'\b(corp|corporation|inc|incorporated|llc|l\.l\.c|ltd|limited|co|company|services|partners|private|pvt)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def clean_vernacular(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def extract_addr_features(addr):
    if not addr: return set(), set()
    norm = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('ASCII').lower()
    norm = re.sub(r'\b(street|st|saint)\b', 'st', norm)
    norm = re.sub(r'\b(road|rd)\b', 'rd', norm)
    norm = re.sub(r'\b(avenue|ave)\b', 'ave', norm)
    norm = re.sub(r'\b(drive|dr)\b', 'dr', norm)
    tokens = re.findall(r'[a-z0-9]+', norm)
    # numbers: strip leading zeros, e.g. 0684 -> 684
    numbers = set()
    for t in tokens:
        if t.isdigit():
            numbers.add(t.lstrip('0') or '0')
    stopwords = {'north','south','east','west','rd','st','ave','dr','suite','floor','near','behind','door','block','sector','hno','plot','flat','shop','null'}
    words = {t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in stopwords}
    return numbers, words

print("Loading 15,000 ground truth rows...")
gt = {}
with open('dataset_folder/student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if len(row) >= 2 and row[1].strip():
            gt[row[0]] = set(row[1].split(','))
            if len(gt) >= 10000:
                break

needed_s1 = set(gt.keys())
needed_targets = {t for s in gt.values() for t in s}

s1_records = {}
with open('dataset_folder/student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if row[0] in needed_s1:
            s1_records[row[0]] = (row[1], row[2], row[3]) # name, addr, country

cand_records = {}
for s_file in ['train_source2.tsv', 'train_source3.tsv']:
    with open('dataset_folder/student_resource/dataset/train/' + s_file, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in needed_targets:
                cand_records[row[0]] = (row[1], row[2], row[3])

print(f"Loaded {len(s1_records)} S1 records, {len(cand_records)} target candidates. Total ground truth matches: {sum(len(v) for v in gt.values())}")

# Let's test blocking key strategies
index = defaultdict(list)

def make_keys(rec_id, name, addr, country):
    keys = set()
    c_name = clean_text(name)
    v_name = clean_vernacular(name)
    nums, words = extract_addr_features(addr)
    
    # 1. Name word keys
    w_list = c_name.split()
    if w_list:
        # First word
        if len(w_list[0]) >= 3:
            keys.add(f"fw:{country}:{w_list[0]}")
        # Two words combined
        if len(w_list) >= 2:
            keys.add(f"nw2:{country}:{w_list[0]}_{w_list[1]}")
        # Exact clean name (if short/distinct)
        keys.add(f"cn:{country}:{c_name[:20]}")
    elif v_name:
        # Vernacular script first 2 tokens
        v_tokens = v_name.split()
        if v_tokens:
            keys.add(f"vn:{country}:{v_tokens[0]}")
            if len(v_tokens) >= 2:
                keys.add(f"vn2:{country}:{v_tokens[0]}_{v_tokens[1]}")
    
    # 2. Address keys (CRUCIAL for cross-lingual and DBA matches)
    for num in nums:
        # Number + country + top distinctive address words
        for w in list(words)[:3]:
            keys.add(f"an:{country}:{num}_{w}")
        # Long numbers (PIN code / phone number >= 5 digits)
        if len(num) >= 5:
            keys.add(f"pin:{country}:{num}")
            
    # 3. Distinctive address word pairs (if no numbers)
    w_sorted = sorted(words)
    if len(w_sorted) >= 2:
        keys.add(f"aw2:{country}:{w_sorted[0]}_{w_sorted[1]}")
        
    return keys

# Build candidate index
for cid, (name, addr, country) in cand_records.items():
    for k in make_keys(cid, name, addr, country):
        index[k].append(cid)

print(f"Index built with {len(index):,} distinct keys.")

# Query S1 records and evaluate recall at top-5, top-10
top5_found = 0
top10_found = 0
total_gt_matches = 0

for s1_id, targets in gt.items():
    name, addr, country = s1_records[s1_id]
    s1_keys = make_keys(s1_id, name, addr, country)
    
    cand_scores = Counter()
    for k in s1_keys:
        for cid in index.get(k, []):
            cand_scores[cid] += 1
            
    top5 = {cid for cid, _ in cand_scores.most_common(5)}
    top10 = {cid for cid, _ in cand_scores.most_common(10)}
    
    for t in targets:
        if t in cand_records: # only test those in our sample
            total_gt_matches += 1
            if t in top5:
                top5_found += 1
            if t in top10:
                top10_found += 1

print("\n==========================================")
print(f"RECALL EVALUATION (Total Sampled Ground Truth: {total_gt_matches:,}):")
print(f"Top-5 Recall:  {top5_found} / {total_gt_matches} = {top5_found / total_gt_matches:.2%}")
print(f"Top-10 Recall: {top10_found} / {total_gt_matches} = {top10_found / total_gt_matches:.2%}")
print("==========================================")
