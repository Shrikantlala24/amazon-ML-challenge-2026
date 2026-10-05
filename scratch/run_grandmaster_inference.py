import os
import sys
import time
import gc
import re
import csv
import math
import unicodedata as ud
from collections import Counter, defaultdict
import numpy as np
import zipfile

from rapidfuzz import fuzz
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio
import lightgbm as lgb

print("🚀 Starting Grandmaster Inference Engine (Country-Decoupled Policies)...", flush=True)

possible_paths = [
    "dataset_folder/student_resource/dataset",
    "student_resource/dataset",
    "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset",
    "/kaggle/input/student-resource/dataset",
    "/content/student_resource/dataset",
    "dataset"
]

DATA_DIR = None
for p in possible_paths:
    if os.path.exists(os.path.join(p, "test", "test_source1.tsv")):
        DATA_DIR = os.path.abspath(p)
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not find test dataset directory!")

TEST_DIR = os.path.join(DATA_DIR, "test")
print("📂 Using TEST_DIR:", TEST_DIR, flush=True)

SUFFIXES = frozenset("inc incorporated corporation corp limited ltd llc llp pvt private plc sarl sas sa co company gmbh".split())
DOMAIN_RE = re.compile(r'\.(com|org|net|in|co\.in|co|biz|io|info|gov|edu)(\b|$)', re.IGNORECASE)
HANDLE_RE = re.compile(r'^[#@]+')

ABBREVIATIONS = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "flr": "floor", "bldg": "building",
    "ste": "sainte", "bvd": "boulevard", "pvt": "private", "ltd": "limited", "dept": "department"
}

def normalize(value):
    value = ud.normalize("NFKC", value or "").casefold()
    clean_chars = [c if ud.category(c)[0] in "LNM" or c in "/" else " " for c in value]
    words = "".join(clean_chars).split()
    expanded = [ABBREVIATIONS.get(w, w) for w in words]
    return " ".join(expanded)

def fold_accents(value):
    out, latin = [], False
    for c in ud.normalize("NFD", value):
        if not ud.category(c).startswith("M"):
            latin = "LATIN" in ud.name(c, "")
        if not (latin and ud.category(c).startswith("M")):
            out.append(c)
    return ud.normalize("NFC", "".join(out))

def clean_web(value):
    if not value: return ""
    t = HANDLE_RE.sub('', value)
    t = DOMAIN_RE.sub(' ', t)
    return " ".join(t.split())

def core_name(value):
    words = clean_web(value).split()
    while words and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words) or clean_web(value) or value

def tokens(value):
    return set(value.split())

def grams(value):
    value = " ".join(value.split())
    return {value[i:i+3] for i in range(max(0, len(value)-2))} if len(value) >= 3 else ({value} if value else set())

def numbers(value):
    return {t for t in value.split() if any(c.isdigit() for c in t)}

def postcodes(value):
    return {t for t in value.split() if t.isdigit() and len(t) in (5, 6)}

def script(value):
    counts = {}
    for c in value:
        if c.isalpha():
            label = ud.name(c, "UNKNOWN").split()[0]
            counts[label] = counts.get(label, 0) + 1
    return max(sorted(counts), key=counts.get) if counts else "NONE"

def doc_views(name, address):
    n, a = normalize(name), normalize(address)
    nf, af = fold_accents(n), fold_accents(a)
    return dict(name=name or "", address=address or "", n=n, a=a, nf=nf, af=af,
                core=core_name(nf), script=script(n))

def doc_keys(doc):
    n, a = doc["nf"], doc["af"]
    if n: yield "N", n
    if doc["core"]: yield "K", doc["core"]
    if a: yield "A", a
    if n and a: yield "X", n + "\x1f" + a
    for t in sorted(tokens(n)):
        if t in tokens(doc["core"]):
            yield "n", t
    for t in sorted(tokens(a)):
        yield "a", t
    for t in sorted(grams(n)):
        yield "g", t
    for p in sorted(postcodes(a)):
        for t in sorted(tokens(doc["core"])):
            yield "p", p + "\x1f" + t
    for num in sorted(numbers(a))[:8]:
        for t in sorted(tokens(doc["core"])):
            yield "d", num + "\x1f" + t

PASS_BITS = {"X": 1, "N": 2, "K": 2, "A": 4, "d": 8, "p": 16, "n": 32, "a": 32, "g": 64}

class SotaChampionIndex:
    def __init__(self, posting_limit=1200, candidate_cap=10, rare_tokens=6, expansion_cap=24000):
        self.posting_limit = posting_limit
        self.candidate_cap = candidate_cap
        self.rare_tokens = rare_tokens
        self.expansion_cap = expansion_cap
        self.docs = {}
        self.postings = defaultdict(list)
        self.df = defaultdict(int)
        self.total_targets = 0
        
    def add_target(self, eid, source, raw_name, raw_addr):
        doc = doc_views(raw_name, raw_addr)
        self.docs[eid] = (doc, source)
        self.total_targets += 1
        seen = set()
        for kind, token in doc_keys(doc):
            key_s = (kind, token, source)
            self.postings[key_s].append(eid)
            if (kind, token) not in seen:
                self.df[(kind, token)] += 1
                seen.add((kind, token))
                
    def idf(self, kind, token):
        df_val = self.df.get((kind, token), 0)
        return math.log1p(self.total_targets / (1 + df_val))
        
    def retrieve(self, s1_doc):
        all_keys = list(doc_keys(s1_doc))
        available = defaultdict(list)
        oversized = 0
        for kind, token in all_keys:
            if kind == "g": continue
            df_val = self.df.get((kind, token), 0)
            if 0 < df_val <= self.posting_limit:
                for source in (2, 3):
                    if (kind, token, source) in self.postings:
                        available[(kind, source)].append((df_val, token, (kind, token, source)))
            elif df_val > self.posting_limit:
                oversized += 1
                
        chosen = []
        for (kind, source), terms in sorted(available.items()):
            terms.sort()
            if kind in ("n", "a"):
                terms = terms[:self.rare_tokens]
            elif kind in ("d", "p"):
                terms = terms[:self.rare_tokens * 2]
            chosen.extend((kind, source, df_val, token, k_s) for df_val, token, k_s in terms)
            
        hits = {}
        expanded = 0
        for kind, source, df_val, token, k_s in sorted(chosen, key=lambda x: (x[2], x[0], x[1], x[3])):
            if expanded + df_val > self.expansion_cap:
                continue
            weight = math.log1p(self.total_targets / (1 + df_val))
            for eid in self.postings[k_s]:
                h = hits.setdefault(eid, {"score": 0.0, "flags": 0, "name_hit": False, "address_hit": False, "min_df": df_val, "source": source})
                h["score"] += weight
                h["flags"] |= PASS_BITS.get(kind, 0)
                h["name_hit"] |= kind in ("n", "N", "K", "X", "d", "p")
                h["address_hit"] |= kind in ("a", "A", "X", "d", "p")
                h["min_df"] = min(h["min_df"], df_val)
            expanded += df_val
            
        rescue = not any(h["name_hit"] and h["address_hit"] for h in hits.values())
        if rescue:
            gram_terms = defaultdict(list)
            for kind, token in all_keys:
                if kind == "g":
                    df_val = self.df.get((kind, token), 0)
                    if 0 < df_val <= self.posting_limit:
                        for source in (2, 3):
                            if (kind, token, source) in self.postings:
                                gram_terms[source].append((df_val, token, (kind, token, source)))
            extra = [("g", s, df_val, t, k_s) for s, terms in gram_terms.items() for df_val, t, k_s in sorted(terms)[:self.rare_tokens * 2]]
            for kind, source, df_val, token, k_s in sorted(extra, key=lambda x: (x[2], x[0], x[1], x[3])):
                if expanded + df_val > self.expansion_cap: continue
                weight = math.log1p(self.total_targets / (1 + df_val))
                for eid in self.postings[k_s]:
                    h = hits.setdefault(eid, {"score": 0.0, "flags": 0, "name_hit": False, "address_hit": False, "min_df": df_val, "source": source})
                    h["score"] += weight
                    h["flags"] |= PASS_BITS.get(kind, 0)
                expanded += df_val
                
        ranked = []
        for eid, h in hits.items():
            target_doc, source = self.docs[eid]
            other_nf = clean_web(target_doc["nf"])
            s1_nf = clean_web(s1_doc["nf"])
            if s1_nf and other_nf:
                r_direct = ratio(s1_nf, other_nf)
                r_sort = token_sort_ratio(s1_nf, other_nf)
                r_set = token_set_ratio(s1_nf, other_nf)
                n_sim = max(r_direct, r_sort, 0.9 * r_set) / 100.0
            else:
                n_sim = 0.0
                
            if s1_doc["af"] and target_doc["af"]:
                a_sim = (0.6 * token_set_ratio(s1_doc["af"], target_doc["af"]) + 0.4 * ratio(s1_doc["af"], target_doc["af"])) / 100.0
            else:
                a_sim = 0.0
                
            h["name_similarity"] = n_sim
            h["address_similarity"] = a_sim
            
            target_has_addr = bool(target_doc["af"])
            s1_has_addr = bool(s1_doc["af"])
            if not target_has_addr or not s1_has_addr:
                cheap = 0.85 * n_sim + 0.15 * min(h["score"] / 50.0, 1.0)
            elif n_sim < 0.20 and a_sim >= 0.65:
                cheap = 0.85 * a_sim + 0.15 * min(h["score"] / 50.0, 1.0)
            else:
                cheap = 0.55 * n_sim + 0.35 * a_sim + 0.10 * min(h["score"] / 50.0, 1.0)
                
            ranked.append((cheap, eid, h, target_doc))
            
        ranked.sort(key=lambda x: (-x[0], x[1]))
        selected_ids = set()
        protected = set()
        for source in (2, 3):
            source_ranked = [r for r in ranked if r[2]['source'] == source]
            selected_ids.update(r[1] for r in source_ranked[:8])
            for field in ('name_similarity', 'address_similarity'):
                view_ranked = sorted(source_ranked, key=lambda r: (-r[2][field], -r[0], r[1]))
                protected.update(r[1] for r in view_ranked[:8] if r[2][field] > 0)
        selected_ids.update(protected)
        selected_ids.update(r[1] for r in ranked if r[2]['flags'] & 63)
        selected = [r for r in ranked if r[1] in selected_ids]
        
        kept = [r for r in selected if r[1] in protected][:self.candidate_cap]
        kept_ids = {r[1] for r in kept}
        kept.extend(r for r in selected if r[1] not in kept_ids)
        kept = sorted(kept[:self.candidate_cap], key=lambda r: (-r[0], r[1]))
        
        out_candidates = []
        for rank_idx, (cheap, eid, h, target_doc) in enumerate(kept, 1):
            out_candidates.append({
                "tid": eid,
                "target": target_doc,
                "retrieval_score": h["score"],
                "cheap_score": cheap,
                "flags": h["flags"],
                "min_df": h.get("min_df", 1),
                "rank": rank_idx,
                "rescue": int(rescue)
            })
        return out_candidates

def overlap(a, b):
    shared = len(a & b)
    return (shared / max(1, len(a | b)), shared / max(1, len(a)), shared / max(1, len(b)))

def compute_pair_features(left, candidate, retriever):
    right = candidate["target"]
    out = {}
    for prefix, field, raw in (("name", "nf", "n"), ("address", "af", "a")):
        x, y = left[field], right[field]
        present = bool(x and y)
        a, b = tokens(x), tokens(y)
        out.update({f"{prefix}_{k}": v for k, v in dict(
            exact=int(present and x == y),
            raw_exact=int(bool(left[raw] and right[raw]) and left[raw] == right[raw]),
            ratio=fuzz.ratio(x, y) / 100 if present else 0,
            token_sort=fuzz.token_sort_ratio(x, y) / 100 if present else 0,
            token_set=fuzz.token_set_ratio(x, y) / 100 if present else 0,
            partial=fuzz.partial_ratio(x, y) / 100 if present else 0,
            length_ratio=min(len(x), len(y)) / max(1, len(x), len(y)),
            left_tokens=len(a), right_tokens=len(b),
            left_missing=int(not x), right_missing=int(not y)).items()})
        for k, v in zip(("jaccard", "left_containment", "right_containment"), overlap(a, b)):
            out[f"{prefix}_{k}"] = v
        out[f"{prefix}_trigram"] = overlap(grams(x), grams(y))[0]
        kind = "n" if prefix == "name" else "a"
        weights = {t: retriever.idf(kind, t) for t in a | b}
        out[f"{prefix}_idf_jaccard"] = sum(weights[t] for t in a & b) / max(1e-9, sum(weights.values()))
        
    a, b = numbers(left["af"]), numbers(right["af"])
    out["number_jaccard"] = overlap(a, b)[0]
    out["number_conflict"] = int(bool(a and b) and not a & b)
    out["number_left_missing"] = int(not a)
    out["number_right_missing"] = int(not b)
    
    a, b = postcodes(left["af"]), postcodes(right["af"])
    out["postcode_agree"] = int(bool(a & b))
    out["postcode_conflict"] = int(bool(a and b) and not a & b)
    
    out["core_ratio"] = fuzz.ratio(left["core"], right["core"]) / 100 if left["core"] and right["core"] else 0
    acronym = lambda v: "".join(w[0] for w in v.split())
    x, y = left["core"], right["core"]
    out["acronym_agree"] = int(bool(x and y) and (acronym(x) == y.replace(" ", "") or acronym(y) == x.replace(" ", "")))
    out["script_agree"] = int(left["script"] != "NONE" and left["script"] == right["script"])
    
    first_num = lambda v: next((t for t in v.split() if any(c.isdigit() for c in t)), "")
    x, y = first_num(left["af"]), first_num(right["af"])
    out["first_number_agree"] = int(bool(x and y) and x == y)
    out["first_number_conflict"] = int(bool(x and y) and x != y)
    
    out["retrieval_score"] = candidate["retrieval_score"]
    out["cheap_score"] = candidate["cheap_score"]
    out["rank"] = candidate["rank"]
    out["min_block_log"] = math.log1p(candidate["min_df"])
    out["pass_count"] = candidate["flags"].bit_count()
    for bit in range(7):
        out[f"pass_{bit}"] = int(bool(candidate["flags"] & (1 << bit)))
    out["rescue"] = candidate["rescue"]
    out["name_address_conflict"] = out["name_ratio"] * (1 - out["address_ratio"])
    out["both_similarity"] = out["name_ratio"] * out["address_ratio"]
    return out

FEATURE_NAMES = [
    'name_exact', 'name_raw_exact', 'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_length_ratio', 'name_left_tokens', 'name_right_tokens', 'name_left_missing', 'name_right_missing',
    'name_jaccard', 'name_left_containment', 'name_right_containment', 'name_trigram', 'name_idf_jaccard',
    'address_exact', 'address_raw_exact', 'address_ratio', 'address_token_sort', 'address_token_set',
    'address_partial', 'address_length_ratio', 'address_left_tokens', 'address_right_tokens',
    'address_left_missing', 'address_right_missing', 'address_jaccard', 'address_left_containment',
    'address_right_containment', 'address_trigram', 'address_idf_jaccard', 'number_jaccard',
    'number_conflict', 'number_left_missing', 'number_right_missing', 'postcode_agree', 'postcode_conflict',
    'core_ratio', 'acronym_agree', 'script_agree', 'first_number_agree', 'first_number_conflict',
    'retrieval_score', 'cheap_score', 'rank', 'min_block_log', 'pass_count', 'pass_0', 'pass_1',
    'pass_2', 'pass_3', 'pass_4', 'pass_5', 'pass_6', 'rescue', 'name_address_conflict', 'both_similarity'
]

COUNTRY_POLICIES = {
    'India':  {'add': 0.535, 'start': 0.575},
    'US':     {'add': 0.635, 'start': 0.650},
    'France': {'add': 0.680, 'start': 0.710}
}

MODEL_PATH = '0.895/code/business_entity_resolution/artifacts/model.txt'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'model.txt'

booster = lgb.Booster(model_file=MODEL_PATH)
print("✅ LightGBM Booster loaded successfully.", flush=True)

COUNTRIES = ['France', 'US', 'India']
os.makedirs("output", exist_ok=True)
CANDIDATES_TSV = "output/candidate_pairs.tsv"
MATCHES_TSV = "output/matching_results.tsv"

with open(CANDIDATES_TSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["source1_entity_id", "candidate_entity_ids"])

with open(MATCHES_TSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["source1_entity_id", "matched_entity_ids"])

total_s1_processed = 0
total_candidates_written = 0
total_matches_predicted = 0
pipeline_start = time.time()

for country in COUNTRIES:
    print("\n" + "=" * 70, flush=True)
    print(f"🌍 STARTING GRANDMASTER STREAMING FOR: {country.upper()}", flush=True)
    print("=" * 70, flush=True)
    c_start = time.time()
    policy = COUNTRY_POLICIES.get(country, {'add': 0.642, 'start': 0.642})
    print(f"  🎯 Calibrated Country Policy: add={policy['add']}, start={policy['start']}", flush=True)
    
    index = SotaChampionIndex(posting_limit=1200, candidate_cap=10, rare_tokens=6, expansion_cap=24000)
    
    n_cands = 0
    for step_num, (src_num, fname) in enumerate([(2, "test_source2.tsv"), (3, "test_source3.tsv")], start=1):
        fpath = os.path.join(TEST_DIR, fname)
        if not os.path.exists(fpath): continue
        print(f"  [Step {step_num}/3] Indexing targets from {fname} for {country}...", flush=True)
        f_t0 = time.time()
        file_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            hdr = next(r)
            c_idx = hdr.index("country")
            for row in r:
                if row[c_idx].strip() == country:
                    eid, name, addr = row[0], row[1], row[2]
                    index.add_target(eid, src_num, name, addr)
                    n_cands += 1
                    file_count += 1
                    if file_count % 400000 == 0:
                        print(f"    ... Scanned {file_count:,} target rows ({time.time()-f_t0:.1f}s)", flush=True)
        print(f"  ✅ Finished {fname}: Added {file_count:,} {country} entities in {time.time()-f_t0:.1f}s.", flush=True)
                    
    print(f"  🎯 [Index Built] Total {n_cands:,} target entities indexed in {time.time()-c_start:.1f}s.", flush=True)
    
    print(f"  [Step 3/3] Scoring & matching test_source1.tsv entities for {country}...", flush=True)
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s1_count = 0
    c_cand_count = 0
    c_match_count = 0
    last_tick = time.time()
    
    with open(s1_path, "r", encoding="utf-8") as f_s1, \
         open(CANDIDATES_TSV, "a", encoding="utf-8", newline="") as f_cand_out, \
         open(MATCHES_TSV, "a", encoding="utf-8", newline="") as f_match_out:
        
        r_s1 = csv.reader(f_s1, delimiter="\t")
        hdr = next(r_s1)
        c_idx = hdr.index("country")
        
        cand_writer = csv.writer(f_cand_out, delimiter="\t")
        match_writer = csv.writer(f_match_out, delimiter="\t")
        
        for row in r_s1:
            if row[c_idx].strip() != country:
                continue
                
            s1_id = row[0]
            s1_name = row[1]
            s1_addr = row[2]
            s1_doc = doc_views(s1_name, s1_addr)
            
            candidates = index.retrieve(s1_doc)
            matched_ids = []
            scored_cands = []
            
            if candidates:
                feature_rows = []
                for cand in candidates:
                    feats = compute_pair_features(s1_doc, cand, index)
                    feature_rows.append([feats[n] for n in FEATURE_NAMES])
                
                x_mat = np.array(feature_rows, dtype=np.float32)
                probs = booster.predict(x_mat)
                
                for idx, cand in enumerate(candidates):
                    feats = compute_pair_features(s1_doc, cand, index)
                    p = float(probs[idx])
                    if feats["first_number_conflict"] and feats["postcode_conflict"]:
                        p *= 0.10
                    scored_cands.append((cand["tid"], p))
                
                scored_cands.sort(key=lambda x: -x[1])
                max_prob = scored_cands[0][1] if scored_cands else 0.0
                
                for cid, p in scored_cands:
                    if p >= policy['add'] and max_prob >= policy['start']:
                        matched_ids.append(cid)
            
            cand_basket = list(matched_ids[:5])
            cand_basket_set = set(cand_basket)
            for cid, _ in scored_cands:
                if len(cand_basket) >= 5: break
                if cid not in cand_basket_set:
                    cand_basket.append(cid)
                    cand_basket_set.add(cid)
                    
            final_matches = [m for m in matched_ids if m in cand_basket_set]
            assert set(final_matches).issubset(set(cand_basket))
            
            c_cand_count += len(cand_basket)
            c_match_count += len(final_matches)
            
            cand_writer.writerow([s1_id, ",".join(cand_basket)])
            match_writer.writerow([s1_id, ",".join(final_matches)])
            
            s1_count += 1
            if s1_count % 25000 == 0:
                elapsed = time.time() - last_tick
                rate = 25000 / elapsed if elapsed > 0 else 0
                print(f"    ⚡ [{country}] {s1_count:,} queries | {c_match_count:,} matches | Rate: {rate:,.0f} q/s", flush=True)
                last_tick = time.time()
                
    total_s1_processed += s1_count
    total_candidates_written += c_cand_count
    total_matches_predicted += c_match_count
    
    print(f"\n  ✅ {country} COMPLETED in {(time.time() - c_start) / 60:.2f} mins.", flush=True)
    print(f"     Entities Processed    : {s1_count:,}", flush=True)
    print(f"     Avg Candidates/Entity : {c_cand_count / s1_count:.2f} (<= 5.0 constraint satisfied)", flush=True)
    print(f"     Matches Predicted     : {c_match_count:,}", flush=True)
    
    del index
    gc.collect()

print(f"\n" + "=" * 70, flush=True)
print(f"🎉 ALL COUNTRIES COMPLETED in {(time.time() - pipeline_start) / 60:.2f} minutes!", flush=True)
print(f"=" * 70, flush=True)
print(f"Total Source 1 Processed: {total_s1_processed:,}", flush=True)
print(f"Overall Avg Candidates  : {total_candidates_written / total_s1_processed:.2f}", flush=True)
print(f"Total Matches Predicted : {total_matches_predicted:,}", flush=True)

# Package final ZIP
zip_name = "submission_v10_grandmaster.zip"
print(f"\n📦 Packaging {zip_name}...", flush=True)
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(CANDIDATES_TSV, arcname="candidate_pairs.tsv")
    zf.write(MATCHES_TSV, arcname="matching_results.tsv")

print(f"🎉 SUCCESS! {zip_name} created ({os.path.getsize(zip_name)/(1024*1024):.1f} MB).", flush=True)
