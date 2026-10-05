import os
import sys
import time
import csv
import re
import math
import unicodedata as ud
from collections import Counter, defaultdict
import zipfile
from rapidfuzz import fuzz

print("🚀 Starting Synthesis of Optimal Matching Results...", flush=True)

DATA_DIR = "dataset_folder/student_resource/dataset"
TEST_DIR = os.path.join(DATA_DIR, "test")

# 1. Multi-View Normalizer
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

def numbers(value):
    return {t for t in value.split() if any(c.isdigit() for c in t)}

def postcodes(value):
    return {t for t in value.split() if t.isdigit() and len(t) in (5, 6)}

def first_num(v):
    return next((t for t in v.split() if any(c.isdigit() for c in t)), "")

# 2. Load Targets Lookup Dictionary into RAM (Compact representation)
print("📦 Loading Target Entity Views (Source 2 & Source 3)...", flush=True)
t0 = time.time()
target_docs = {}  # eid -> (nf, af, core, nums, pcode, fnum)

for fname in ["test_source2.tsv", "test_source3.tsv"]:
    fpath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(fpath): continue
    print(f"  Reading {fname}...", flush=True)
    with open(fpath, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        hdr = next(r)
        c_idx = hdr.index("country")
        for row in r:
            eid, name, addr = row[0], row[1], row[2]
            n, a = normalize(name), normalize(addr)
            nf, af = fold_accents(n), fold_accents(a)
            core = core_name(nf)
            target_docs[eid] = (nf, af, core, numbers(af), postcodes(af), first_num(af))

print(f"✅ Loaded {len(target_docs):,} targets in {time.time()-t0:.1f}s.", flush=True)

# 3. Stream Source 1 Queries and Apply Country-Decoupled Precision Filter
s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
cands_in_path = "0.895/output/candidate_pairs.tsv"
match_in_path = "0.895/output/matching_results.tsv"

os.makedirs("output", exist_ok=True)
cands_out_path = "output/candidate_pairs.tsv"
match_out_path = "output/matching_results.tsv"

print("🎬 Processing & Refining Match Predictions...", flush=True)
t_start = time.time()
total_rows = 0
total_cands = 0
total_matches = 0
violations = 0
last_tick = time.time()

with open(s1_path, "r", encoding="utf-8") as fs1, \
     open(cands_in_path, "r", encoding="utf-8") as fc_in, \
     open(match_in_path, "r", encoding="utf-8") as fm_in, \
     open(cands_out_path, "w", encoding="utf-8", newline="") as fc_out, \
     open(match_out_path, "w", encoding="utf-8", newline="") as fm_out:
     
    rs1 = csv.reader(fs1, delimiter="\t")
    rc = csv.reader(fc_in, delimiter="\t")
    rm = csv.reader(fm_in, delimiter="\t")
    
    wc = csv.writer(fc_out, delimiter="\t")
    wm = csv.writer(fm_out, delimiter="\t")
    
    hdr_s1 = next(rs1)
    hdr_c = next(rc)
    hdr_m = next(rm)
    
    wc.writerow(["source1_entity_id", "candidate_entity_ids"])
    wm.writerow(["source1_entity_id", "matched_entity_ids"])
    
    for row_s1, row_c, row_m in zip(rs1, rc, rm):
        s1_id = row_s1[0]
        s1_name = row_s1[1]
        s1_addr = row_s1[2]
        country = row_s1[3].strip()
        
        s1_n, s1_a = normalize(s1_name), normalize(s1_addr)
        s1_nf, s1_af = fold_accents(s1_n), fold_accents(s1_a)
        s1_core = core_name(s1_nf)
        s1_nums = numbers(s1_af)
        s1_pcodes = postcodes(s1_af)
        s1_fnum = first_num(s1_af)
        
        cands_raw = row_c[1].split(",") if row_c[1] else []
        matches_raw = row_m[1].split(",") if row_m[1] else []
        
        # Refined scoring for candidates
        refined_scored = []
        candidate_pool = list(dict.fromkeys(matches_raw + cands_raw[:15]))
        
        for cid in candidate_pool:
            t_info = target_docs.get(cid)
            if not t_info:
                # Target not in S2/S3
                continue
            t_nf, t_af, t_core, t_nums, t_pcodes, t_fnum = t_info
            
            # Similarities
            n_ratio = fuzz.ratio(s1_nf, t_nf) / 100.0
            n_sort = fuzz.token_sort_ratio(s1_nf, t_nf) / 100.0
            n_set = fuzz.token_set_ratio(s1_nf, t_nf) / 100.0
            n_sim = max(n_ratio, n_sort, 0.9 * n_set)
            
            core_ratio = fuzz.ratio(s1_core, t_core) / 100.0 if s1_core and t_core else 0.0
            
            if s1_af and t_af:
                a_sim = (0.6 * fuzz.token_set_ratio(s1_af, t_af) + 0.4 * fuzz.ratio(s1_af, t_af)) / 100.0
            else:
                a_sim = 0.0
                
            pcode_conflict = bool(s1_pcodes and t_pcodes and not (s1_pcodes & t_pcodes))
            fnum_conflict = bool(s1_fnum and t_fnum and s1_fnum != t_fnum)
            
            # Base quality score
            if not s1_af or not t_af:
                score = 0.85 * max(n_sim, core_ratio)
            elif n_sim < 0.20 and a_sim >= 0.70:
                score = 0.85 * a_sim
            else:
                score = 0.55 * n_sim + 0.35 * a_sim + 0.10 * core_ratio
                
            # Hard vetoes
            if fnum_conflict and pcode_conflict:
                score *= 0.10
            elif country == "France" and fnum_conflict and a_sim < 0.60:
                score *= 0.20
            elif country == "US" and pcode_conflict:
                score *= 0.25
                
            # Existing match bonus (priors from LightGBM)
            if cid in matches_raw:
                score += 0.15
                
            refined_scored.append((cid, score))
            
        refined_scored.sort(key=lambda x: -x[1])
        
        # Select matches based on Country Policy
        matched = []
        if country == "France":
            thresh = 0.70
        elif country == "India":
            thresh = 0.58
        else: # US
            thresh = 0.65
            
        max_s = refined_scored[0][1] if refined_scored else 0.0
        if max_s >= thresh:
            for cid, s in refined_scored:
                if s >= thresh and len(matched) < 5:
                    matched.append(cid)
                    
        # Build strict Top 5 Basket containing all matches
        basket = list(matched[:5])
        basket_set = set(basket)
        for cid, _ in refined_scored:
            if len(basket) >= 5: break
            if cid not in basket_set:
                basket.append(cid)
                basket_set.add(cid)
                
        final_matches = [m for m in matched if m in basket_set]
        assert set(final_matches).issubset(set(basket))
        assert len(basket) <= 5
        
        wc.writerow([s1_id, ",".join(basket)])
        wm.writerow([s1_id, ",".join(final_matches)])
        
        total_rows += 1
        total_cands += len(basket)
        total_matches += len(final_matches)
        
        if total_rows % 250000 == 0:
            elapsed = time.time() - last_tick
            rate = 250000 / elapsed if elapsed > 0 else 0
            print(f"  ⚡ Processed {total_rows:,} / 1,732,544 rows ({time.time()-t_start:.1f}s) | Rate: {rate:,.0f} rows/s", flush=True)
            last_tick = time.time()

print("\n" + "=" * 65, flush=True)
print(f"🎉 Synthesis Completed in {(time.time()-t_start)/60:.2f} minutes!", flush=True)
print(f"Total Rows Processed : {total_rows:,} (Expected: 1,732,544)", flush=True)
print(f"Average Candidates   : {total_cands / total_rows:.2f} (<= 5.0 constraint satisfied)", flush=True)
print(f"Total Matches        : {total_matches:,} (Avg: {total_matches/total_rows:.2f})", flush=True)
print("=" * 65, flush=True)

# Package into submission_synthesized_0.999.zip
zip_path = "submission_synthesized_0.999.zip"
print(f"\n📦 Packaging {zip_path}...", flush=True)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(cands_out_path, arcname="candidate_pairs.tsv")
    zf.write(match_out_path, arcname="matching_results.tsv")

print(f"🎉 SUCCESS! {zip_path} created ({os.path.getsize(zip_path)/(1024*1024):.1f} MB).", flush=True)
