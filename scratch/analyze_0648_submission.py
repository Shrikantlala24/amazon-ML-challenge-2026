import csv
from collections import Counter

def analyze_submission(tsv_path):
    total = 0
    empty = 0
    match_counts = Counter()
    total_matches = 0
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        hdr = next(r)
        for row in r:
            total += 1
            if len(row) < 2 or not row[1].strip():
                empty += 1
                match_counts[0] += 1
            else:
                m = row[1].split(",")
                cnt = len(m)
                match_counts[cnt] += 1
                total_matches += cnt
                
    print(f"=== File: {tsv_path} ===")
    print(f"Total Source 1 entities: {total:,}")
    print(f"Predicted Empty (Singletons): {empty:,} ({empty/total:.2%})")
    print(f"Predicted Non-Empty: {total - empty:,} ({(total - empty)/total:.2%})")
    print(f"Total Matches Predicted: {total_matches:,}")
    print(f"Avg Matches per Non-Empty: {total_matches / (total - empty) if (total - empty) > 0 else 0:.2f}")
    print("Match Count Distribution:")
    for k in sorted(match_counts.keys())[:10]:
        print(f"  {k} matches: {match_counts[k]:,} ({match_counts[k]/total:.2%})")
    print()

analyze_submission("0.648/matching_results.tsv")
analyze_submission("matching_results.tsv") # baseline 0.610
