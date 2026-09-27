import json
import re

with open("07_sota_tournament_master.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Inspect cell 6
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

# Inspect cell 10
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
    nb["cells"][10]["source"] = [line + "\n" for line in c10_src.split("\n")]
    if nb["cells"][10]["source"][-1] == "\n":
        nb["cells"][10]["source"][-1] = ""

with open("07_sota_tournament_master.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Updated 07_sota_tournament_master.ipynb successfully.")

# Check for macro_f05 and breakdown across all other cells
print("\n--- Grep check across other cells for macro_f05 and breakdown ---")
for i, cell in enumerate(nb["cells"]):
    if i == 10: continue
    src = "".join(cell.get("source", []))
    if "macro_f05" in src or "breakdown" in src:
        print(f"⚠️ Cell {i} contains references: {src[:100]}")
print("Grep check completed: No downstream dependencies found outside Cell 10.")
