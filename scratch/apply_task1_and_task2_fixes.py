import json
import os

with open("07_sota_tournament_master.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 6 (Index): Add make_keys wrapper before the print statement
c6_src = "".join(nb["cells"][6]["source"])
wrapper_code = '''def make_keys(raw_name, raw_addr, country):
    """Standalone wrapper so non-class callers (e.g. the validation harness
    in Cell 10) can call make_keys without instantiating a full index."""
    _tmp = SotaInvertedIndex.__new__(SotaInvertedIndex)
    return SotaInvertedIndex.make_keys(_tmp, raw_name, raw_addr, country)

print("✅ SotaInvertedIndex ready.")'''

if 'def make_keys(raw_name, raw_addr, country):' not in c6_src:
    c6_src = c6_src.replace('print("✅ SotaInvertedIndex ready.")', wrapper_code)
    nb["cells"][6]["source"] = [line + "\n" for line in c6_src.split("\n")]
    if nb["cells"][6]["source"][-1] == "\n":
        nb["cells"][6]["source"][-1] = ""

# 2. Update Cell 10 (Validation Harness): Gate run_validation behind RUN_VALIDATION flag
c10_src = "".join(nb["cells"][10]["source"])
old_val_call = "macro_f05, breakdown = run_validation()"
new_val_call = """RUN_VALIDATION = False  # Set True only when re-validating after a code
                         # change. Leave False for actual submission runs —
                         # validation runs on train data and isn't needed to
                         # produce matching_results.tsv / candidate_pairs.tsv.

if RUN_VALIDATION:
    macro_f05, breakdown = run_validation()
else:
    print("⏭️  Skipping 30k validation harness (RUN_VALIDATION=False) — "
          "proceeding directly to test-set inference in the next section.")"""

if old_val_call in c10_src:
    c10_src = c10_src.replace(old_val_call, new_val_call)
elif "RUN_VALIDATION = False" not in c10_src:
    c10_src = c10_src.rstrip() + "\n\n" + new_val_call + "\n"

nb["cells"][10]["source"] = [line + "\n" for line in c10_src.split("\n")]
if nb["cells"][10]["source"][-1] == "\n":
    nb["cells"][10]["source"][-1] = ""

# Save 07_sota_tournament_master.ipynb
with open("07_sota_tournament_master.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Successfully updated 07_sota_tournament_master.ipynb.")
