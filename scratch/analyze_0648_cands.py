import csv
from collections import Counter

total = 0
empty = 0
cand_counts = Counter()
total_cands = 0

with open("0.648/candidate_pairs.tsv", "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    hdr = next(r)
    for row in r:
        total += 1
        if len(row) < 2 or not row[1].strip():
            empty += 1
            cand_counts[0] += 1
        else:
            c = row[1].split(",")
            cnt = len(c)
            cand_counts[cnt] += 1
            total_cands += cnt

print(f"=== Candidate Pairs Analysis: 0.648/candidate_pairs.tsv ===")
print(f"Total entities: {total:,}")
print(f"Entities with 0 candidates: {empty:,} ({empty/total:.2%})")
print(f"Entities with >=1 candidates: {total - empty:,} ({(total - empty)/total:.2%})")
print(f"Total candidates in baskets: {total_cands:,}")
print(f"Average candidates per entity: {total_cands / total:.2f}")
print("Candidate Count Distribution:")
for k in sorted(cand_counts.keys()):
    print(f"  {k} candidates: {cand_counts[k]:,} ({cand_counts[k]/total:.2%})")
