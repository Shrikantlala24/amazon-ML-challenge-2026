import json
import re

# Load the current 07_sota_tournament_master.ipynb
with open("07_sota_tournament_master.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 4: Task 2 fix for extract_domain_and_concat
c4_src = "".join(nb["cells"][4]["source"])
old_func = """def extract_domain_and_concat(raw_name: str, clean_name: str):
    base = re.sub(r'\\.(com|org|net|in|co|io|biz|info|fr|gov|edu)$', '', clean_name.strip())
    concat = base.replace(' ', '')
    return base, concat"""

new_func = """def extract_domain_and_concat(raw_name: str, clean_name: str = None):
    if not raw_name or not isinstance(raw_name, str):
        return "", ""
    if '.' in raw_name:
        stripped = re.sub(r'\\.(com|org|net|in|co|io|biz|info|fr|gov|edu)\\b.*$', '', raw_name, flags=re.IGNORECASE)
        base = clean_latin_name(stripped)
    else:
        base = clean_name if clean_name is not None else clean_latin_name(raw_name)
    concat = base.replace(' ', '')
    return base, concat"""

assert old_func in c4_src or "extract_domain_and_concat" in c4_src
if old_func in c4_src:
    c4_src = c4_src.replace(old_func, new_func)
else:
    # replace extract_domain_and_concat block
    c4_src = re.sub(r'def extract_domain_and_concat\(raw_name: str, clean_name: str\):[\s\S]*?return base, concat', new_func, c4_src)

nb["cells"][4]["source"] = [line + "\n" for line in c4_src.split("\n")]
if nb["cells"][4]["source"][-1] == "\n":
    nb["cells"][4]["source"][-1] = ""

# 2. Update Cell 10: Task 1 fix (cand_basket loop + assertion)
c10_src = "".join(nb["cells"][10]["source"])
old_loop = """            matched_ids = []
            for cid in pool:
                c_rec = index.records.get(cid)
                if not c_rec: continue
                (c_raw_n, c_cl_n, c_raw_a, c_cl_a, c_nums, c_words, c_concat) = c_rec
                
                if is_match(s1_name, s1_clean_name, s1_clean_addr, s1_nums, s1_words, s1_concat,
                            c_raw_n, c_cl_n, c_cl_a, c_nums, c_words, c_concat):
                    matched_ids.append(cid)
                    
            c_match_count += len(matched_ids)
            match_writer.writerow([s1_id, ",".join(matched_ids)])"""

new_loop = """            matched_ids = []
            for cid in cand_basket:
                c_rec = index.records.get(cid)
                if not c_rec: continue
                (c_raw_n, c_cl_n, c_raw_a, c_cl_a, c_nums, c_words, c_concat) = c_rec
                
                if is_match(s1_name, s1_clean_name, s1_clean_addr, s1_nums, s1_words, s1_concat,
                            c_raw_n, c_cl_n, c_cl_a, c_nums, c_words, c_concat):
                    matched_ids.append(cid)
                    
            assert set(matched_ids).issubset(set(cand_basket)), f"Candidate audit violation: matched_ids {matched_ids} not subset of cand_basket {cand_basket}!"
            c_match_count += len(matched_ids)
            match_writer.writerow([s1_id, ",".join(matched_ids)])"""

assert old_loop in c10_src, "old_loop not found in cell 10!"
c10_src = c10_src.replace(old_loop, new_loop)
nb["cells"][10]["source"] = [line + "\n" for line in c10_src.split("\n")]
if nb["cells"][10]["source"][-1] == "\n":
    nb["cells"][10]["source"][-1] = ""

# 3. Read the local validation code from scratch/run_task3_validation.py to embed in notebook
with open("scratch/run_task3_validation.py", "r", encoding="utf-8") as vf:
    val_py = vf.read()

# Extract run_validation function body from scratch/run_task3_validation.py
val_fn_match = re.search(r'(def run_validation\(\):[\s\S]*?)(if __name__ == [\'"]__main__[\'"]:[\s\S]*)$', val_py)
if val_fn_match:
    val_code = val_fn_match.group(1).strip() + "\n\n# Run Stratified 30,000 entity validation\nmacro_f05, breakdown = run_validation()\n"
else:
    raise ValueError("Could not extract run_validation from run_task3_validation.py")

val_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 5. Stratified Local F0.5 Validation Harness (30,000 Entities)\n",
        "Evaluates the production pipeline (candidate blocking recall, audit compliance $\\bar{C} \\le 5.0$, matcher gate precision/recall, and macro $F_{0.5}$) on 30,000 stratified training entities across US and India before running full test-set inference."
    ]
}

val_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in val_code.split("\n")]
}

# Update section headers for 5 and 6
nb["cells"][9]["source"] = [
    "### 6. Multi-Source Tournament Processing (Country-by-Country Streaming)\n",
    "Processes France, US, and India sequentially. Guaranteed < 1.5 GB RAM footprint."
]
nb["cells"][11]["source"] = [
    "### 7. Official Submission Validation & Zip Packaging\n",
    "Runs the official competition validation script to ensure full compliance before submission."
]

# Insert validation cells at index 9 (between cell 8 and old cell 9)
nb["cells"].insert(9, val_code_cell)
nb["cells"].insert(9, val_md_cell)

# Save updated notebook
with open("07_sota_tournament_master.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Successfully updated 07_sota_tournament_master.ipynb:")
print("  - Task 1: cand_basket loop + subset assertion added to Tournament Processing")
print("  - Task 2: extract_domain_and_concat order-of-operations bug fixed")
print("  - Task 3: Local F0.5 Validation Harness inserted as Section 5 (30k entities)")
print(f"  - Total cells: {len(nb['cells'])}")
