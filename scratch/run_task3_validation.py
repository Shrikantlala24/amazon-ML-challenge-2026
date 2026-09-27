import sys
import os
import time
import re
import csv
import unicodedata
from collections import defaultdict, Counter
from rapidfuzz import fuzz

DATA_DIR = "dataset_folder/student_resource/dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")

# 1. Normalizers
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
    if not raw_name or not isinstance(raw_name, str):
        return "", ""
    stripped = re.sub(r'\.(com|org|net|in|co|io|biz|info|fr|gov|edu)\b.*$', '', raw_name, flags=re.IGNORECASE)
    base = clean_latin_name(stripped)
    concat = base.replace(' ', '')
    return base, concat

ROAD_MAP = {
    r'\b(street|str|saint)\b': 'st', r'\b(road)\b': 'rd', r'\b(avenue|ave\.)\b': 'ave',
    r'\b(boulevard|blvd\.)\b': 'blvd', r'\b(drive|dr\.)\b': 'dr', r'\b(lane|ln\.)\b': 'ln',
    r'\b(highway|hwy\.)\b': 'hwy', r'\b(rue|r\.)\b': 'rue', r'\b(allee|all\.)\b': 'allee'
}
STOPWORDS = {'north','south','east','west','rd','st','ave','dr','suite','floor','near','behind','door','block','sector','hno','plot','flat','shop','null','city','township','state','district','box','po'}

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

# 2. Key Generation
def make_keys(raw_name, raw_addr, country):
    keys = set()
    c_name = clean_latin_name(raw_name)
    v_name = clean_vernacular_name(raw_name)
    base_dom, concat = extract_domain_and_concat(raw_name, c_name)
    nums, words = extract_addr_tokens(raw_addr)
    
    if len(concat) >= 4:
        keys.add(f"cc:{country}:{concat[:8]}")
        keys.add(f"cc5:{country}:{concat[:5]}")
        
    w_list = c_name.split()
    if w_list:
        if len(w_list[0]) >= 3:
            keys.add(f"fw:{country}:{w_list[0]}")
        if len(w_list) >= 2:
            keys.add(f"w2:{country}:{w_list[0]}_{w_list[1]}")
            keys.add(f"init:{country}:{w_list[0][0]}{w_list[1]}")
        elif len(w_list) == 1:
            keys.add(f"init:{country}:{w_list[0]}")
            
        if len(c_name) <= 22:
            keys.add(f"cn:{country}:{c_name}")
            
        for w in w_list:
            if len(w) >= 6:
                keys.add(f"ng3:{country}:{w[:3]}_{w[-3:]}")
    elif v_name:
        v_tokens = v_name.split()
        if v_tokens:
            keys.add(f"vn:{country}:{v_tokens[0]}")
            if len(v_tokens) >= 2:
                keys.add(f"vn2:{country}:{v_tokens[0]}_{v_tokens[1]}")
                
    w_list_sorted = sorted(list(words))
    for num in nums:
        for w in w_list_sorted[:3]:
            keys.add(f"an:{country}:{num}_{w}")
        if len(num) >= 5:
            keys.add(f"pin:{country}:{num}")
            
    for i in range(min(3, len(w_list_sorted))):
        for j in range(i + 1, min(4, len(w_list_sorted))):
            keys.add(f"aw2:{country}:{w_list_sorted[i]}_{w_list_sorted[j]}")
            
    return keys

# 3. Matcher Gate
def is_match(s1_raw_name, s1_clean_name, s1_clean_addr, s1_nums, s1_words, s1_concat,
             cand_raw_name, cand_clean_name, cand_clean_addr, cand_nums, cand_words, cand_concat) -> bool:
    n1 = s1_clean_name
    n2 = cand_clean_name
    
    # 1. Exact or High-Confidence Name Match
    if n1 and n2:
        if n1 == n2:
            return True
        sort_sim = fuzz.token_sort_ratio(n1, n2)
        if sort_sim >= 84:
            return True
        set_sim = fuzz.token_set_ratio(n1, n2)
        if set_sim >= 88:
            return True
            
        # 2. Domain / Substring Concatenation Match
        c1 = s1_concat
        c2 = cand_concat
        if c1 and c2 and (c1 in c2 or c2 in c1) and len(min(c1, c2)) >= 5:
            if not s1_clean_addr or not cand_clean_addr:
                return True
            addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
            if addr_sim >= 55:
                return True
                
        # 2b. Initial + Word Match (e.g. z + blue = zblue)
        w1_toks = n1.split()
        w2_toks = n2.split()
        init1 = (w1_toks[0][0] + w1_toks[1]) if len(w1_toks) >= 2 else ""
        init2 = (w2_toks[0][0] + w2_toks[1]) if len(w2_toks) >= 2 else ""
        if (init1 and (init1 == c2 or init1 in c2)) or (init2 and (init2 == c1 or init2 in c1)):
            if not s1_clean_addr or not cand_clean_addr:
                return True
            if fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr) >= 55:
                return True
                
        # 3. Moderate Name Similarity (>= 68) with Supporting Address
        if sort_sim >= 68 or set_sim >= 75:
            if s1_clean_addr and cand_clean_addr:
                addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
                if addr_sim >= 65:
                    return True
                if (s1_nums & cand_nums) and (s1_words & cand_words):
                    return True
                    
    # 4. Cross-Lingual & DBA Address Match (Same address units / PIN codes)
    if s1_clean_addr and cand_clean_addr:
        common_nums = s1_nums & cand_nums
        common_words = s1_words & cand_words
        addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
        
        # House number match + locality/street words or high address similarity
        if common_nums and len(common_words) >= 1 and (addr_sim >= 70 or len(common_words) >= 2):
            return True
            
        # Shared PIN code or phone number (>= 5 digits)
        for num in common_nums:
            if len(num) >= 5:
                return True
                
        # Strong address token overlap (addresses identical despite different name/DBA)
        if addr_sim >= 82 and len(common_words) >= 2:
            return True
            
    return False

def run_validation():
    print("=" * 70)
    print("📊 Amazon ML Challenge 2026: Local F0.5 Validation Harness (30,000 Sample)")
    print("=" * 70)
    print("⚠️  DATASET LIMITATION NOTE:")
    print("    France is present ONLY in the test dataset (~15% of test records).")
    print("    The training dataset contains only US (~60%) and India (~40%).")
    print("    Therefore, this validation harness evaluates the US and India partitions.")
    print("=" * 70)

    # 1. Stratified Sampling of 30,000 S1 records from train_source1.tsv
    sample_targets_us = 18000
    sample_targets_in = 12000
    
    sampled_s1 = {'US': [], 'India': []}
    s1_needed_ids = set()
    
    print("\n[Step 1/5] Sampling 30,000 stratified entities from train_source1.tsv...")
    t0 = time.time()
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
                
    print(f"  Sampled {len(sampled_s1['US']):,} US entities and {len(sampled_s1['India']):,} India entities in {time.time()-t0:.2f}s.")
    
    # 2. Load Ground Truth for Sampled Entities
    print("\n[Step 2/5] Loading ground truth for sampled entities...")
    t0 = time.time()
    gt = {}
    total_true_matches = 0
    singletons = 0
    
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1_id = row[0]
            if s1_id in s1_needed_ids:
                if len(row) >= 2 and row[1].strip():
                    targets = set(row[1].split(','))
                    gt[s1_id] = targets
                    total_true_matches += len(targets)
                else:
                    gt[s1_id] = set()
                    singletons += 1
                if len(gt) >= len(s1_needed_ids):
                    break
                    
    print(f"  Loaded ground truth in {time.time()-t0:.2f}s: {len(gt):,} entities, {total_true_matches:,} true target links, {singletons:,} singletons ({singletons/len(gt):.2%}).")

    # 3. Country-by-Country Production Execution
    results_by_country = {}
    all_s1_f05 = []
    all_candidates_count = 0
    total_blocking_hits = 0
    total_cand_basket_hits = 0
    total_gt_non_singleton = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    failing_examples = []
    
    for country in ['US', 'India']:
        print(f"\n=================================================================")
        print(f"🌍 Running Validation on {country} ({len(sampled_s1[country]):,} S1 entities)...")
        print(f"=================================================================")
        c_t0 = time.time()
        
        # Build inverted index for this country
        print(f"  [A] Indexing target records from train_source2.tsv and train_source3.tsv...")
        index = defaultdict(list)
        idx_t0 = time.time()
        n_targets = 0
        
        # We index targets from train_source2 and train_source3
        for fname in ["train_source2.tsv", "train_source3.tsv"]:
            fpath = os.path.join(TRAIN_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                hdr = next(r)
                c_idx = hdr.index("country")
                for row in r:
                    if row[c_idx].strip() == country:
                        eid = row[0]
                        name = row[1]
                        addr = row[2]
                        for k in make_keys(name, addr, country):
                            index[k].append(eid)
                        n_targets += 1
                        
        print(f"      Indexed {n_targets:,} target records in {time.time()-idx_t0:.2f}s ({len(index):,} distinct keys).")
        
        # [B] Query candidates for sampled S1 entities
        print(f"  [B] Querying candidates for {len(sampled_s1[country]):,} {country} entities...")
        q_t0 = time.time()
        
        s1_query_data = [] # (s1_id, s1_name, s1_addr, pool, cand_basket)
        needed_target_ids = set()
        
        for s1_id, s1_name, s1_addr in sampled_s1[country]:
            keys = make_keys(s1_name, s1_addr, country)
            counts = Counter()
            for k in keys:
                for eid in index.get(k, []):
                    counts[eid] += 1
                    
            pool = [eid for eid, _ in counts.most_common(8)]
            cand_basket = pool[:5] # Strict Task 1 fix
            
            s1_query_data.append((s1_id, s1_name, s1_addr, pool, cand_basket))
            for cid in cand_basket:
                needed_target_ids.add(cid)
                
        print(f"      Queried in {time.time()-q_t0:.2f}s. Unique target records to evaluate: {len(needed_target_ids):,}")
        
        # Free index memory
        del index
        
        # [C] Load attributes only for the retrieved candidates in cand_basket
        print(f"  [C] Loading attributes for {len(needed_target_ids):,} retrieved candidates...")
        attr_t0 = time.time()
        target_attributes = {}
        for fname in ["train_source2.tsv", "train_source3.tsv"]:
            fpath = os.path.join(TRAIN_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r)
                for row in r:
                    eid = row[0]
                    if eid in needed_target_ids:
                        raw_name = row[1]
                        raw_addr = row[2]
                        c_name = clean_latin_name(raw_name)
                        c_addr = clean_address(raw_addr)
                        nums, words = extract_addr_tokens(raw_addr)
                        _, concat = extract_domain_and_concat(raw_name, c_name)
                        target_attributes[eid] = (raw_name, c_name, c_addr, nums, words, concat)
                        if len(target_attributes) >= len(needed_target_ids):
                            break
                            
        print(f"      Attributes loaded in {time.time()-attr_t0:.2f}s.")
        
        # [D] Run Matcher Gate over cand_basket ONLY (Task 1 logic) and compute metrics
        print(f"  [D] Running Matcher Gate over cand_basket only & evaluating F0.5...")
        m_t0 = time.time()
        
        c_f05_scores = []
        c_cand_counts = 0
        c_blocking_pool_hits = 0
        c_cand_basket_hits = 0
        c_gt_matches_count = 0
        c_tp = 0
        c_fp = 0
        c_fn = 0
        
        for s1_id, s1_name, s1_addr, pool, cand_basket in s1_query_data:
            c_cand_counts += len(cand_basket)
            true_targets = gt.get(s1_id, set())
            c_gt_matches_count += len(true_targets)
            
            # Blocking recall metrics
            c_blocking_pool_hits += len(set(pool) & true_targets)
            c_cand_basket_hits += len(set(cand_basket) & true_targets)
            
            s1_cl_name = clean_latin_name(s1_name)
            s1_cl_addr = clean_address(s1_addr)
            s1_nums, s1_words = extract_addr_tokens(s1_addr)
            _, s1_concat = extract_domain_and_concat(s1_name, s1_cl_name)
            
            # Matcher evaluation over cand_basket ONLY
            matched_ids = []
            for cid in cand_basket:
                c_rec = target_attributes.get(cid)
                if not c_rec: continue
                (c_raw_n, c_cl_n, c_cl_a, c_nums, c_words, c_concat) = c_rec
                
                if is_match(s1_name, s1_cl_name, s1_cl_addr, s1_nums, s1_words, s1_concat,
                            c_raw_n, c_cl_n, c_cl_a, c_nums, c_words, c_concat):
                    matched_ids.append(cid)
                    
            # Task 1 subset assertion check
            assert set(matched_ids).issubset(set(cand_basket)), f"Subset violation for {s1_id}!"
            
            # Per-entity F0.5 calculation
            tp = len(set(matched_ids) & true_targets)
            fp = len(set(matched_ids) - true_targets)
            fn = len(true_targets - set(matched_ids))
            
            c_tp += tp
            c_fp += fp
            c_fn += fn
            
            if len(true_targets) == 0:
                # Singleton entity
                f05 = 1.0 if len(matched_ids) == 0 else 0.0
            elif tp == 0:
                f05 = 0.0
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                
            c_f05_scores.append(f05)
            all_s1_f05.append(f05)
            
            # Record failing examples if F0.5 < 0.5 and true_targets exist
            if f05 < 0.5 and len(true_targets) > 0 and len(failing_examples) < 8:
                failing_examples.append({
                    'country': country,
                    's1_id': s1_id,
                    's1_name': s1_name,
                    's1_addr': s1_addr,
                    'true_targets': true_targets,
                    'pool': pool,
                    'cand_basket': cand_basket,
                    'matched_ids': matched_ids,
                    'tp': tp, 'fp': fp, 'fn': fn, 'f05': f05
                })
                
        all_candidates_count += c_cand_counts
        total_blocking_hits += c_blocking_pool_hits
        total_cand_basket_hits += c_cand_basket_hits
        total_gt_non_singleton += c_gt_matches_count
        total_tp += c_tp
        total_fp += c_fp
        total_fn += c_fn
        
        c_macro_f05 = sum(c_f05_scores) / len(c_f05_scores)
        c_avg_cands = c_cand_counts / len(sampled_s1[country])
        c_pool_recall = c_blocking_pool_hits / c_gt_matches_count if c_gt_matches_count > 0 else 0
        c_basket_recall = c_cand_basket_hits / c_gt_matches_count if c_gt_matches_count > 0 else 0
        c_gate_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        c_gate_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        
        results_by_country[country] = {
            'macro_f05': c_macro_f05,
            'avg_candidates': c_avg_cands,
            'blocking_recall_pool8': c_pool_recall,
            'blocking_recall_basket5': c_basket_recall,
            'matcher_precision': c_gate_prec,
            'matcher_recall': c_gate_rec,
            'tp': c_tp, 'fp': c_fp, 'fn': c_fn,
            'time_sec': time.time() - c_t0
        }
        
        print(f"  ✅ {country} Completed in {(time.time()-c_t0)/60:.2f} mins:")
        print(f"     Macro F0.5 Score         : {c_macro_f05:.4f} ({c_macro_f05*100:.2f}%)")
        print(f"     Average Candidates/Entity: {c_avg_cands:.2f} (Audit constraint <= 5.0: {'PASS' if c_avg_cands <= 5.0 else 'FAIL'})")
        print(f"     Blocking Recall (Top 8)  : {c_pool_recall:.2%}")
        print(f"     Blocking Recall (Top 5)  : {c_basket_recall:.2%}")
        print(f"     Matcher Gate Precision   : {c_gate_prec:.2%}")
        print(f"     Matcher Gate Recall      : {c_gate_rec:.2%}")

    # Overall Summary
    macro_f05_overall = sum(all_s1_f05) / len(all_s1_f05)
    overall_avg_cands = all_candidates_count / len(all_s1_f05)
    overall_pool_recall = total_blocking_hits / total_gt_non_singleton if total_gt_non_singleton > 0 else 0
    overall_basket_recall = total_cand_basket_hits / total_gt_non_singleton if total_gt_non_singleton > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    print("\n" + "=" * 70)
    print("🏆 OVERALL VALIDATION SUMMARY (30,000 Stratified S1 Entities)")
    print("=" * 70)
    print(f"Overall Macro F0.5 Score       : {macro_f05_overall:.4f} ({macro_f05_overall*100:.2f}%)")
    print(f"Average Candidates per Entity  : {overall_avg_cands:.2f} (<= 5.0 Audit Requirement: {'PASSED ✅' if overall_avg_cands <= 5.0 else 'FAILED ❌'})")
    print(f"Candidate Blocking Recall (Pool 8): {overall_pool_recall:.2%}")
    print(f"Candidate Blocking Recall (Top 5) : {overall_basket_recall:.2%}")
    print(f"Overall Matcher Gate Precision : {overall_prec:.2%}")
    print(f"Overall Matcher Gate Recall    : {overall_rec:.2%}")
    print("-" * 70)
    print("PER-COUNTRY BREAKDOWN:")
    for c, r in results_by_country.items():
        print(f"  {c:7s}: Macro F0.5 = {r['macro_f05']:.4f} | Avg Cands = {r['avg_candidates']:.2f} | Blocking Rec(Top 5) = {r['blocking_recall_basket5']:.2%} | Prec = {r['matcher_precision']:.2%} | Rec = {r['matcher_recall']:.2%}")
    print("=" * 70)

    # Failing examples
    if len(failing_examples) > 0:
        print("\n🔍 SAMPLE FAILING PAIRS AUDIT (Up to 5 examples):")
        for i, ex in enumerate(failing_examples[:5]):
            print(f"\n--- Failing Example #{i+1} [{ex['country']}] ---")
            print(f"  S1 ID: {ex['s1_id']}")
            print(f"  S1 Name: '{ex['s1_name']}'")
            print(f"  S1 Addr: '{ex['s1_addr']}'")
            print(f"  True Target IDs: {list(ex['true_targets'])}")
            print(f"  Top 5 Cand Basket: {ex['cand_basket']}")
            print(f"  Predicted Matches: {ex['matched_ids']}")
            print(f"  TP: {ex['tp']}, FP: {ex['fp']}, FN: {ex['fn']} -> Entity F0.5: {ex['f05']:.4f}")
            in_pool = set(ex['pool']) & ex['true_targets']
            in_basket = set(ex['cand_basket']) & ex['true_targets']
            print(f"  Target in Top 8 Pool: {list(in_pool) if in_pool else 'MISS (Blocking Recall failure)'}")
            print(f"  Target in Top 5 Basket: {list(in_basket) if in_basket else 'MISS (Ranked 6-8 or not retrieved)'}")
            if in_basket and not ex['matched_ids']:
                print(f"  Failure Mode: MATCHER PRECISION/THRESHOLD TOO STRICT (Candidate was in basket but rejected by is_match)")
            elif not in_basket:
                print(f"  Failure Mode: BLOCKING / CANDIDATE GENERATION MISS")
                
    return macro_f05_overall, results_by_country

if __name__ == '__main__':
    run_validation()
