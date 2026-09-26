import csv

total = 0
empty = 0
matches_count = 0
with open("matching_results.tsv", "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        total += 1
        if len(row) < 2 or not row[1].strip():
            empty += 1
        else:
            m = row[1].split(",")
            matches_count += len(m)

print(f"Total S1 entities: {total}")
print(f"Predicted empty (singletons): {empty} ({empty/total:.1%})")
print(f"Predicted non-empty: {total - empty} ({(total - empty)/total:.1%})")
print(f"Total matches predicted: {matches_count} (average {matches_count/(total-empty) if total-empty > 0 else 0:.2f} per non-empty)")
