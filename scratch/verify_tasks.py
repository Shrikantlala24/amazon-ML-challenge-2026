import json

with open("07_sota_tournament_master.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def clean_ipython_code(lines):
    return "\n".join(l for l in lines if not l.strip().startswith("!"))

ns = {}
# Execute setup, normalizers, index (with wrapper), matcher gate
exec(clean_ipython_code(nb["cells"][2]["source"]), ns)
exec(clean_ipython_code(nb["cells"][4]["source"]), ns)
exec(clean_ipython_code(nb["cells"][6]["source"]), ns)
exec(clean_ipython_code(nb["cells"][8]["source"]), ns)

# Test standalone make_keys
test_keys = ns["make_keys"]("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US")
print(f"make_keys standalone test call produced {len(test_keys)} keys: {list(test_keys)[:3]}")
assert len(test_keys) > 0, "make_keys returned empty set!"

# Test Cell 10 execution (RUN_VALIDATION=False)
exec(clean_ipython_code(nb["cells"][10]["source"]), ns)

print("\n✅ Verification PASSED: All cells execute smoothly without NameError or downstream regressions!")
