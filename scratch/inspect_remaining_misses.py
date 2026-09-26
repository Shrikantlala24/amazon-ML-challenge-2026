import os
import csv
import re
import unicodedata
from collections import defaultdict, Counter

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_tournament_recall import (
    clean_name, clean_address, extract_numbers, make_tournament_keys,
    gt, s1_records, cand_records, index
)

missed = []
for s1_id, targets in gt.items():
    if not targets: continue
    s1_rname, s1_raddr = s1_records[s1_id]
    cn = clean_name(s1_rname)
    ca = clean_address(s1_raddr)
    num = extract_numbers(s1_raddr)
    
    keys = make_tournament_keys(cn, ca, num)
    counts = Counter()
    for k in keys:
        for cid in index.get(k, []):
            counts[cid] += 1
    found = {cid for cid, _ in counts.most_common(6)}
    
    for t in targets:
        if t in cand_records and t not in found:
            t_rname, t_raddr = cand_records[t]
            missed.append((s1_rname, s1_raddr, t_rname, t_raddr))

print(f"Remaining missed: {len(missed)}")
print("--- 8 Remaining Missed Examples ---")
for i, (s1_n, s1_a, c_n, c_a) in enumerate(missed[:8]):
    print(f"\nExample #{i+1}:")
    print(f"  S1   : '{s1_n}' | '{s1_a}'")
    print(f"  Cand : '{c_n}' | '{c_a}'")
    print(f"  Clean S1 : '{clean_name(s1_n)}' | '{clean_address(s1_a)}'")
    print(f"  Clean Can: '{clean_name(c_n)}' | '{clean_address(c_a)}'")
