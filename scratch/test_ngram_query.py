import time
from collections import defaultdict, Counter

# Test retrieval logic
index = defaultdict(list)
records = {}

def get_char_3grams(text):
    text = f" {text.lower()} "
    return {text[i:i+3] for i in range(len(text)-2)}

# Add some records
records["cand_1"] = "Christ Chapel Church"
records["cand_2"] = "McDonalds Fast Food Restaurant"
records["cand_3"] = "Blue Bottle Coffee"

for cid, text in records.items():
    for ng in get_char_3grams(text):
        index[ng].append(cid)

# Query with typo: "Christ Chape1" (OCR typo: 1 instead of l)
query = "Christ Chape1"
q_ngs = get_char_3grams(query)
counts = Counter()
for ng in q_ngs:
    for cid in index.get(ng, []):
        counts[cid] += 1

print("Query:", query)
for cid, cnt in counts.most_common(3):
    jaccard = cnt / (len(q_ngs) + len(get_char_3grams(records[cid])) - cnt)
    print(f"Match: {cid} ({records[cid]}) -> {cnt} overlapping 3-grams, Jaccard: {jaccard:.2f}")
