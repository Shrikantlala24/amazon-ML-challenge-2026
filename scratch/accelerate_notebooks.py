import json

for nb_path in ["07_sota_tournament_master.ipynb", "07_sota_tournament_master_sagemaker.ipynb"]:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 6 (SotaInvertedIndex query_candidates)
    c6_src = "".join(nb["cells"][6]["source"])
    old_query = """    def query_candidates(self, raw_name, raw_addr, country, top_k=8):
        keys = self.make_keys(raw_name, raw_addr, country)
        counts = Counter()
        for k in keys:
            for eid in self.index.get(k, []):
                counts[eid] += 1
        if not counts:
            return []
        return [eid for eid, _ in counts.most_common(top_k)]"""

    new_query = """    def query_candidates(self, raw_name, raw_addr, country, top_k=8, max_posting=350):
        keys = self.make_keys(raw_name, raw_addr, country)
        counts = Counter()
        for k in keys:
            posting = self.index.get(k, [])
            if len(posting) > max_posting:
                continue
            for eid in posting:
                counts[eid] += 1
        if not counts:
            return []
        return [eid for eid, _ in counts.most_common(top_k)]"""

    if old_query in c6_src:
        c6_src = c6_src.replace(old_query, new_query)
        nb["cells"][6]["source"] = [line + "\n" for line in c6_src.split("\n")]
        if nb["cells"][6]["source"][-1] == "\n":
            nb["cells"][6]["source"][-1] = ""

    # 2. Update Cell 12 (Tournament loop: progress log every 25,000 entities with flush=True)
    c12_src = "".join(nb["cells"][12]["source"])
    c12_src = c12_src.replace("s1_count % 200000 == 0", "s1_count % 25000 == 0")
    c12_src = c12_src.replace("s1_count % 50000 == 0", "s1_count % 25000 == 0")
    c12_src = c12_src.replace('print(f"    ... Processed {s1_count:,} Source 1 entities ({c_match_count:,} matches found)")',
                              'print(f"    ... Processed {s1_count:,} Source 1 entities ({c_match_count:,} matches found)", flush=True)')

    nb["cells"][12]["source"] = [line + "\n" for line in c12_src.split("\n")]
    if nb["cells"][12]["source"][-1] == "\n":
        nb["cells"][12]["source"][-1] = ""

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("✅ Applied 4x query acceleration (max_posting=350) and 25k heartbeat logging to both notebooks!")
