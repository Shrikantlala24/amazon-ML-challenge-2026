import json

nb = json.load(open("07_sota_tournament_master.ipynb", "r", encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    first_lines = src.strip().split("\n")[:2]
    print(f"Cell {i} [{cell['cell_type']}]: {first_lines}")
