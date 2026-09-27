import sys
import os
import time
import re
import csv
import gc
import unicodedata
from collections import defaultdict, Counter
from rapidfuzz import fuzz

DATA_DIR = "dataset_folder/student_resource/dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")

TLD_REGEX = re.compile(r'\.(com|org|net|in|co|io|biz|info|fr|gov|edu)\b.*$', flags=re.IGNORECASE)
ROAD_MAP = {
    r'\b(street|str|saint)\b': 'st', r'\b(road)\b': 'rd', r'\b(avenue|ave\.)\b': 'ave',
    r'\b(boulevard|blvd\.)\b': 'blvd', r'\b(drive|dr\.)\b': 'dr', r'\b(lane|ln\.)\b': 'ln',
    r'\b(highway|hwy\.)\b': 'hwy', r'\b(rue|r\.)\b': 'rue', r'\b(allee|all\.)\b': 'allee'
}
STOPWORDS = {'north','south','east','west','rd','st','ave','dr','suite','floor','near','behind','door','block','sector','hno','plot','flat','shop','null','city','township','state','district','box','po'}

def clean_latin_name(name: str) -> str:
    if not name or not isinstance(name, str): return ""
    text = unicodedata.normalize('NFKD', name)
    text = text.encode('ASCII', 'ignore').decode('ASCII').lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'\b(d\.?b\.?a|trading as|t/a)\b.*', ' ', text)
    text = re.sub(r'\b(corp|corporation|inc|incorporated|llc|l\.l\.c|ltd|limited|co|company|services|partners|private|pvt|sarl|sas|sasu|eurl|sci)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def clean_vernacular_name(name: str) -> str:
    if not name or not isinstance(name, str): return ""
    text = name.lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def extract_domain_and_concat(raw_name: str, clean_name: str = None):
    if not raw_name or not isinstance(raw_name, str): return "", ""
    if '.' in raw_name:
        stripped = TLD_REGEX.sub('', raw_name)
        if stripped != raw_name:
            base = clean_latin_name(stripped)
        else:
            base = clean_name if clean_name is not None else clean_latin_name(raw_name)
    else:
        base = clean_name if clean_name is not None else clean_latin_name(raw_name)
    concat = base.replace(' ', '')
    return base, concat

def clean_address(addr: str) -> str:
    if not addr or not isinstance(addr, str): return ""
    norm = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('ASCII').lower()
    for pat, rep in ROAD_MAP.items():
        norm = re.sub(pat, rep, norm)
    norm = re.sub(r'[^a-z0-9\s,]', ' ', norm)
    return ' '.join(norm.split())

def extract_addr_tokens(addr: str):
    if not addr or not isinstance(addr, str): return set(), set()
    norm = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('ASCII').lower()
    for pat, rep in ROAD_MAP.items():
        norm = re.sub(pat, rep, norm)
    tokens = re.findall(r'[a-z0-9]+', norm)
    numbers = set()
    for t in tokens:
        if t.isdigit():
            numbers.add(t.lstrip('0') or '0')
    words = {t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in STOPWORDS}
    return numbers, words

def make_keys_cleaned(c_name, v_name, concat, nums, words, country):
    keys = set()
    if len(concat) >= 4:
        keys.add(f"cc:{country}:{concat[:8]}")
        keys.add(f"cc5:{country}:{concat[:5]}")
    w_list = c_name.split()
    if w_list:
        if len(w_list[0]) >= 3: keys.add(f"fw:{country}:{w_list[0]}")
        if len(w_list) >= 2:
            keys.add(f"w2:{country}:{w_list[0]}_{w_list[1]}")
            keys.add(f"init:{country}:{w_list[0][0]}{w_list[1]}")
        elif len(w_list) == 1: keys.add(f"init:{country}:{w_list[0]}")
        if len(c_name) <= 22: keys.add(f"cn:{country}:{c_name}")
        for w in w_list:
            if len(w) >= 6: keys.add(f"ng3:{country}:{w[:3]}_{w[-3:]}")
    elif v_name:
        v_tokens = v_name.split()
        if v_tokens:
            keys.add(f"vn:{country}:{v_tokens[0]}")
            if len(v_tokens) >= 2: keys.add(f"vn2:{country}:{v_tokens[0]}_{v_tokens[1]}")
    w_list_sorted = sorted(list(words))
    for num in nums:
        for w in w_list_sorted[:3]: keys.add(f"an:{country}:{num}_{w}")
        if len(num) >= 5: keys.add(f"pin:{country}:{num}")
    for i in range(min(3, len(w_list_sorted))):
        for j in range(i + 1, min(4, len(w_list_sorted))):
            keys.add(f"aw2:{country}:{w_list_sorted[i]}_{w_list_sorted[j]}")
    return keys

def is_match(s1_raw_name, s1_clean_name, s1_clean_addr, s1_nums, s1_words, s1_concat,
             cand_raw_name, cand_clean_name, cand_clean_addr, cand_nums, cand_words, cand_concat) -> bool:
    n1 = s1_clean_name
    n2 = cand_clean_name
    if n1 and n2:
        if n1 == n2: return True
        sort_sim = fuzz.token_sort_ratio(n1, n2)
        if sort_sim >= 84: return True
        set_sim = fuzz.token_set_ratio(n1, n2)
        if set_sim >= 88: return True
        c1 = s1_concat
        c2 = cand_concat
        if c1 and c2 and (c1 in c2 or c2 in c1) and len(min(c1, c2)) >= 5:
            if not s1_clean_addr or not cand_clean_addr: return True
            if fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr) >= 55: return True
        w1_toks = n1.split()
        w2_toks = n2.split()
        init1 = (w1_toks[0][0] + w1_toks[1]) if len(w1_toks) >= 2 else ""
        init2 = (w2_toks[0][0] + w2_toks[1]) if len(w2_toks) >= 2 else ""
        if (init1 and (init1 == c2 or init1 in c2)) or (init2 and (init2 == c1 or init2 in c1)):
            if not s1_clean_addr or not cand_clean_addr: return True
            if fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr) >= 55: return True
        if sort_sim >= 68 or set_sim >= 75:
            if s1_clean_addr and cand_clean_addr:
                if fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr) >= 65: return True
                if (s1_nums & cand_nums) and (s1_words & cand_words): return True
    if s1_clean_addr and cand_clean_addr:
        common_nums = s1_nums & cand_nums
        common_words = s1_words & cand_words
        addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
        if common_nums and len(common_words) >= 1 and (addr_sim >= 70 or len(common_words) >= 2): return True
        for num in common_nums:
            if len(num) >= 5: return True
        if addr_sim >= 82 and len(common_words) >= 2: return True
    return False

# Sample 30,000 S1 records
sample_targets_us = 18000
sample_targets_in = 12000
sampled_s1 = {'US': [], 'India': []}
s1_needed_ids = set()

with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    hdr = next(r)
    c_idx = hdr.index("country")
    for row in r:
        country = row[c_idx].strip()
        if country == 'US' and len(sampled_s1['US']) < sample_targets_us:
            sampled_s1['US'].append((row[0], row[1], row[2]))
            s1_needed_ids.add(row[0])
        elif country == 'India' and len(sampled_s1['India']) < sample_targets_in:
            sampled_s1['India'].append((row[0], row[1], row[2]))
            s1_needed_ids.add(row[0])
        if len(sampled_s1['US']) >= sample_targets_us and len(sampled_s1['India']) >= sample_targets_in:
            break

# Load Ground Truth
gt = {}
with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row[0] in s1_needed_ids:
            gt[row[0]] = set(row[1].split(',')) if len(row) >= 2 and row[1].strip() else set()
            if len(gt) >= len(s1_needed_ids): break

# Precompute query features and keys (Single-Clean)
s1_features = {'US': [], 'India': []}
s1_needed_keys = {'US': set(), 'India': set()}

for country in ['US', 'India']:
    for s1_id, s1_name, s1_addr in sampled_s1[country]:
        c_n = clean_latin_name(s1_name)
        v_n = clean_vernacular_name(s1_name)
        c_a = clean_address(s1_addr)
        nums, words = extract_addr_tokens(s1_addr)
        _, concat = extract_domain_and_concat(s1_name, c_n)
        keys = make_keys_cleaned(c_n, v_n, concat, nums, words, country)
        s1_features[country].append((s1_id, s1_name, c_n, v_n, s1_addr, c_a, nums, words, concat, keys))
        for k in keys:
            s1_needed_keys[country].add(k)

# Index target records (Single-Clean)
index = {'US': defaultdict(list), 'India': defaultdict(list)}
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = os.path.join(TRAIN_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        hdr = next(r)
        c_idx = hdr.index("country")
        for row in r:
            country = row[c_idx].strip()
            if country in index:
                raw_name, raw_addr = row[1], row[2]
                c_n = clean_latin_name(raw_name)
                v_n = clean_vernacular_name(raw_name)
                nums, words = extract_addr_tokens(raw_addr)
                _, concat = extract_domain_and_concat(raw_name, c_n)
                for k in make_keys_cleaned(c_n, v_n, concat, nums, words, country):
                    if k in s1_needed_keys[country]:
                        index[country][k].append(row[0])

# Collect all candidates retrieved across both configurations to load attributes once
configs = [
    ("NO CAP (Confirmed Baseline)", None),
    ("max_posting=350 (Unfiltered Cap)", 350),
    ("Selective Cap (cap only generic ng3/aw2 > 350)", "selective")
]

needed_target_ids = set()
queries_by_config = {}

for label, cap_val in configs:
    s1_queries = {'US': [], 'India': []}
    for country in ['US', 'India']:
        for item in s1_features[country]:
            s1_id, raw_n, c_n, v_n, raw_a, c_a, nums, words, concat, keys = item
            counts = Counter()
            for k in keys:
                post = index[country].get(k, [])
                if cap_val is not None:
                    if cap_val == "selective":
                        # Only cap noisy generic 3-grams and address word pairs
                        if (k.startswith("ng3:") or k.startswith("aw2:") or k.startswith("fw:")) and len(post) > 350:
                            continue
                    elif len(post) > cap_val:
                        continue
                for eid in post:
                    counts[eid] += 1
            pool = [eid for eid, _ in counts.most_common(8)]
            cand_basket = pool[:5]
            s1_queries[country].append((s1_id, raw_n, c_n, c_a, nums, words, concat, pool, cand_basket))
            for cid in cand_basket:
                needed_target_ids.add(cid)
    queries_by_config[label] = s1_queries

# Load attributes for all needed candidates
target_attributes = {}
for fname in ["train_source2.tsv", "train_source3.tsv"]:
    fpath = os.path.join(TRAIN_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            eid = row[0]
            if eid in needed_target_ids:
                raw_name, raw_addr = row[1], row[2]
                c_name = clean_latin_name(raw_name)
                c_addr = clean_address(raw_addr)
                nums, words = extract_addr_tokens(raw_addr)
                _, concat = extract_domain_and_concat(raw_name, c_name)
                target_attributes[eid] = (raw_name, c_name, c_addr, nums, words, concat)
                if len(target_attributes) >= len(needed_target_ids): break
    if len(target_attributes) >= len(needed_target_ids): break

print("\n" + "=" * 80)
print("📊 TASK 2: EVALUATION OF POSTING CAP IMPACT (30,000 STRATIFIED S1 ENTITIES)")
print("=" * 80)

for label, s1_queries in queries_by_config.items():
    country_res = {}
    all_f05 = []
    all_cands = 0
    tot_pool_hits = 0
    tot_basket_hits = 0
    tot_gt = 0
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    
    for country in ['US', 'India']:
        c_f05 = []
        c_cands = 0
        c_pool_hits = 0
        c_basket_hits = 0
        c_gt = 0
        c_tp, c_fp, c_fn = 0, 0, 0
        
        for s1_id, raw_n, c_n, c_a, nums, words, concat, pool, cand_basket in s1_queries[country]:
            c_cands += len(cand_basket)
            true_t = gt.get(s1_id, set())
            c_gt += len(true_t)
            c_pool_hits += len(set(pool) & true_t)
            c_basket_hits += len(set(cand_basket) & true_t)
            
            matched = []
            for cid in cand_basket:
                rec = target_attributes.get(cid)
                if not rec: continue
                if is_match(raw_n, c_n, c_a, nums, words, concat,
                            rec[0], rec[1], rec[2], rec[3], rec[4], rec[5]):
                    matched.append(cid)
            assert set(matched).issubset(set(cand_basket))
            
            tp = len(set(matched) & true_t)
            fp = len(set(matched) - true_t)
            fn = len(true_t - set(matched))
            c_tp += tp
            c_fp += fp
            c_fn += fn
            
            if len(true_t) == 0:
                f05 = 1.0 if len(matched) == 0 else 0.0
            elif tp == 0:
                f05 = 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f05 = (1.25 * p * r) / (0.25 * p + r)
            c_f05.append(f05)
            all_f05.append(f05)
            
        country_res[country] = {
            'f05': sum(c_f05) / len(c_f05),
            'avg_cands': c_cands / len(s1_queries[country]),
            'basket_rec': c_basket_hits / c_gt if c_gt > 0 else 0,
            'pool_rec': c_pool_hits / c_gt if c_gt > 0 else 0,
            'prec': c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0,
            'rec': c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        }
        all_cands += c_cands
        tot_pool_hits += c_pool_hits
        tot_basket_hits += c_basket_hits
        tot_gt += c_gt
        tot_tp += c_tp
        tot_fp += c_fp
        tot_fn += c_fn

    overall_f05 = sum(all_f05) / len(all_f05)
    overall_basket_rec = tot_basket_hits / tot_gt
    overall_prec = tot_tp / (tot_tp + tot_fp)
    overall_rec = tot_tp / (tot_tp + tot_fn)
    overall_cands = all_cands / 30000
    
    print(f"\nConfiguration: {label}")
    print(f"  Overall Macro F0.5 : {overall_f05:.4f} ({overall_f05*100:.2f}%)")
    print(f"  Avg Candidates/Ent : {overall_cands:.2f}")
    print(f"  Blocking Rec (Top5): {overall_basket_rec:.2%}")
    print(f"  Matcher Precision  : {overall_prec:.2%}")
    print(f"  Matcher Recall     : {overall_rec:.2%}")
    print(f"  US Breakdown       : F0.5={country_res['US']['f05']:.4f} | BasketRec={country_res['US']['basket_rec']:.2%} | Prec={country_res['US']['prec']:.2%} | Rec={country_res['US']['rec']:.2%}")
    print(f"  India Breakdown    : F0.5={country_res['India']['f05']:.4f} | BasketRec={country_res['India']['basket_rec']:.2%} | Prec={country_res['India']['prec']:.2%} | Rec={country_res['India']['rec']:.2%}")

print("\n" + "=" * 80)
