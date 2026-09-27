import json

for nb_path in ["07_sota_tournament_master.ipynb", "07_sota_tournament_master_sagemaker.ipynb"]:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 6 (SotaInvertedIndex refactoring: make_keys takes cleaned inputs, query_candidates without cap)
    c6_src = """class SotaInvertedIndex:
    def __init__(self, max_candidates=5):
        self.index = defaultdict(list)
        self.records = {}
        self.max_candidates = max_candidates
        
    def make_keys(self, c_name, v_name, concat, nums, words, country):
        keys = set()
        
        # 1. Concatenated Name & Domain Prefixes (catches zblue.com, truefactory.com)
        if len(concat) >= 4:
            keys.add(f"cc:{country}:{concat[:8]}")
            keys.add(f"cc5:{country}:{concat[:5]}")
            
        # 2. Latin Word Keys & Initials (handles Zander Blue -> zblue)
        w_list = c_name.split()
        if w_list:
            if len(w_list[0]) >= 3:
                keys.add(f"fw:{country}:{w_list[0]}")
            if len(w_list) >= 2:
                keys.add(f"w2:{country}:{w_list[0]}_{w_list[1]}")
                # Initial + next word concatenated (e.g. z + blue = zblue)
                keys.add(f"init:{country}:{w_list[0][0]}{w_list[1]}")
            elif len(w_list) == 1:
                keys.add(f"init:{country}:{w_list[0]}")
                
            if len(c_name) <= 22:
                keys.add(f"cn:{country}:{c_name}")
                
            # Character 3-Gram keys on distinctive words (catches typos: enterpires -> enterprises)
            for w in w_list:
                if len(w) >= 6:
                    keys.add(f"ng3:{country}:{w[:3]}_{w[-3:]}")
                    
        # 3. Vernacular Script Keys (Indian languages: Hindi, Tamil, Telugu)
        elif v_name:
            v_tokens = v_name.split()
            if v_tokens:
                keys.add(f"vn:{country}:{v_tokens[0]}")
                if len(v_tokens) >= 2:
                    keys.add(f"vn2:{country}:{v_tokens[0]}_{v_tokens[1]}")
                    
        # 4. Address & Numeric Anchor Keys (Essential for cross-lingual & DBA entities)
        w_list_sorted = sorted(list(words))
        for num in nums:
            for w in w_list_sorted[:3]:
                keys.add(f"an:{country}:{num}_{w}")
            # PIN code or phone number (>= 5 digits)
            if len(num) >= 5:
                keys.add(f"pin:{country}:{num}")
                
        # Locality / Distinctive Street Word Pairs (top 4 tokens)
        for i in range(min(3, len(w_list_sorted))):
            for j in range(i + 1, min(4, len(w_list_sorted))):
                keys.add(f"aw2:{country}:{w_list_sorted[i]}_{w_list_sorted[j]}")
            
        return keys

    def add_record(self, eid, raw_name, raw_addr, country):
        c_name = clean_latin_name(raw_name)
        v_name = clean_vernacular_name(raw_name)
        c_addr = clean_address(raw_addr)
        nums, words = extract_addr_tokens(raw_addr)
        base_dom, concat = extract_domain_and_concat(raw_name, c_name)
        
        self.records[eid] = (raw_name, c_name, raw_addr, c_addr, nums, words, concat)
        for k in self.make_keys(c_name, v_name, concat, nums, words, country):
            self.index[k].append(eid)

    def query_candidates(self, c_name, v_name, concat, nums, words, country, top_k=8):
        keys = self.make_keys(c_name, v_name, concat, nums, words, country)
        counts = Counter()
        for k in keys:
            for eid in self.index.get(k, []):
                counts[eid] += 1
        if not counts:
            return []
        return [eid for eid, _ in counts.most_common(top_k)]

def make_keys(raw_name, raw_addr, country):
    \"\"\"Standalone wrapper accepting raw strings, extracting features once, and calling index key generation.\"\"\"
    c_name = clean_latin_name(raw_name)
    v_name = clean_vernacular_name(raw_name)
    nums, words = extract_addr_tokens(raw_addr)
    _, concat = extract_domain_and_concat(raw_name, c_name)
    _tmp = SotaInvertedIndex.__new__(SotaInvertedIndex)
    return SotaInvertedIndex.make_keys(_tmp, c_name, v_name, concat, nums, words, country)

print("✅ SotaInvertedIndex ready.")"""

    nb["cells"][6]["source"] = [line + "\n" for line in c6_src.split("\n")]
    if nb["cells"][6]["source"][-1] == "\n":
        nb["cells"][6]["source"][-1] = ""

    # 2. Update Cell 12 (Tournament Loop: clean features once and pass to query_candidates)
    c12_src = "".join(nb["cells"][12]["source"])
    
    # Replace query_candidates call and cleaning block
    old_c12_block = """            # Retrieve top 8 candidates from index
            pool = index.query_candidates(s1_name, s1_addr, country, top_k=8)
            
            # Candidate Baskets: strictly capped at Top 5 (Amazon audit requirement)
            cand_basket = pool[:5]
            c_cand_count += len(cand_basket)
            cand_writer.writerow([s1_id, ",".join(cand_basket)])
            
            # Matcher Evaluation: evaluate candidates for true positive prediction
            s1_clean_name = clean_latin_name(s1_name)
            s1_clean_addr = clean_address(s1_addr)
            s1_nums, s1_words = extract_addr_tokens(s1_addr)
            _, s1_concat = extract_domain_and_concat(s1_name, s1_clean_name)"""

    new_c12_block = """            # Clean Source 1 features ONCE
            s1_clean_name = clean_latin_name(s1_name)
            s1_vernacular = clean_vernacular_name(s1_name)
            s1_clean_addr = clean_address(s1_addr)
            s1_nums, s1_words = extract_addr_tokens(s1_addr)
            _, s1_concat = extract_domain_and_concat(s1_name, s1_clean_name)
            
            # Retrieve top 8 candidates using pre-cleaned features
            pool = index.query_candidates(s1_clean_name, s1_vernacular, s1_concat, s1_nums, s1_words, country, top_k=8)
            
            # Candidate Baskets: strictly capped at Top 5 (Amazon audit requirement)
            cand_basket = pool[:5]
            c_cand_count += len(cand_basket)
            cand_writer.writerow([s1_id, ",".join(cand_basket)])"""

    if old_c12_block in c12_src:
        c12_src = c12_src.replace(old_c12_block, new_c12_block)
    else:
        # Generic replacement if spacing differs
        c12_src = c12_src.replace("pool = index.query_candidates(s1_name, s1_addr, country, top_k=8)",
                                  "s1_clean_name = clean_latin_name(s1_name)\n            s1_vernacular = clean_vernacular_name(s1_name)\n            s1_clean_addr = clean_address(s1_addr)\n            s1_nums, s1_words = extract_addr_tokens(s1_addr)\n            _, s1_concat = extract_domain_and_concat(s1_name, s1_clean_name)\n            pool = index.query_candidates(s1_clean_name, s1_vernacular, s1_concat, s1_nums, s1_words, country, top_k=8)")
        # remove redundant duplicate clean
        dup_clean = """            # Matcher Evaluation: evaluate candidates for true positive prediction
            s1_clean_name = clean_latin_name(s1_name)
            s1_clean_addr = clean_address(s1_addr)
            s1_nums, s1_words = extract_addr_tokens(s1_addr)
            _, s1_concat = extract_domain_and_concat(s1_name, s1_clean_name)"""
        c12_src = c12_src.replace(dup_clean, "            # Matcher Evaluation: evaluate candidates for true positive prediction")

    nb["cells"][12]["source"] = [line + "\n" for line in c12_src.split("\n")]
    if nb["cells"][12]["source"][-1] == "\n":
        nb["cells"][12]["source"][-1] = ""

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("✅ Updated both 07_sota_tournament_master.ipynb and 07_sota_tournament_master_sagemaker.ipynb with Task 1 single-clean refactor!")
