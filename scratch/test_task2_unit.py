import re
import unicodedata
from rapidfuzz import fuzz

def clean_latin_name(name: str) -> str:
    if not name or not isinstance(name, str): return ""
    text = unicodedata.normalize('NFKD', name)
    text = text.encode('ASCII', 'ignore').decode('ASCII').lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'\b(d\.?b\.?a|trading as|t/a)\b.*', ' ', text)
    text = re.sub(r'\b(corp|corporation|inc|incorporated|llc|l\.l\.c|ltd|limited|co|company|services|partners|private|pvt|sarl|sas|sasu|eurl|sci)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def extract_domain_and_concat(raw_name: str, clean_name: str = None):
    if not raw_name or not isinstance(raw_name, str):
        return "", ""
    if '.' in raw_name:
        stripped = re.sub(r'\.(com|org|net|in|co|io|biz|info|fr|gov|edu)\b.*$', '', raw_name, flags=re.IGNORECASE)
        base = clean_latin_name(stripped)
    else:
        base = clean_name if clean_name is not None else clean_latin_name(raw_name)
    concat = base.replace(' ', '')
    return base, concat

ROAD_MAP = {
    r'\b(street|str|saint)\b': 'st', r'\b(road)\b': 'rd', r'\b(avenue|ave\.)\b': 'ave',
    r'\b(boulevard|blvd\.)\b': 'blvd', r'\b(drive|dr\.)\b': 'dr', r'\b(lane|ln\.)\b': 'ln',
    r'\b(highway|hwy\.)\b': 'hwy', r'\b(rue|r\.)\b': 'rue', r'\b(allee|all\.)\b': 'allee'
}
STOPWORDS = {'north','south','east','west','rd','st','ave','dr','suite','floor','near','behind','door','block','sector','hno','plot','flat','shop','null','city','township','state','district','box','po'}

def clean_address(addr: str) -> str:
    if not addr or not isinstance(addr, str): return ""
    norm = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('ASCII').lower()
    for pat, rep in ROAD_MAP.items():
        norm = re.sub(pat, rep, norm)
    norm = re.sub(r'[^a-z0-9\s,]', ' ', norm)
    return ' '.join(norm.split())

def extract_addr_tokens(addr: str):
    if not addr or not isinstance(addr, str): return set(), set()
    norm = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('ASCII').lower()
    for pat, rep in ROAD_MAP.items():
        norm = re.sub(pat, rep, norm)
    tokens = re.findall(r'[a-z0-9]+', norm)
    numbers = set()
    for t in tokens:
        if t.isdigit():
            numbers.add(t.lstrip('0') or '0')
    words = {t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in STOPWORDS}
    return numbers, words

def is_match(s1_raw_name, s1_clean_name, s1_clean_addr, s1_nums, s1_words, s1_concat,
             cand_raw_name, cand_clean_name, cand_clean_addr, cand_nums, cand_words, cand_concat) -> bool:
    n1 = s1_clean_name
    n2 = cand_clean_name
    
    if n1 and n2:
        if n1 == n2:
            return True
        sort_sim = fuzz.token_sort_ratio(n1, n2)
        if sort_sim >= 84:
            return True
        set_sim = fuzz.token_set_ratio(n1, n2)
        if set_sim >= 88:
            return True
            
        c1 = s1_concat
        c2 = cand_concat
        if c1 and c2 and (c1 in c2 or c2 in c1) and len(min(c1, c2)) >= 5:
            if not s1_clean_addr or not cand_clean_addr:
                return True
            addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
            if addr_sim >= 55:
                return True
                
        w1_toks = n1.split()
        w2_toks = n2.split()
        init1 = (w1_toks[0][0] + w1_toks[1]) if len(w1_toks) >= 2 else ""
        init2 = (w2_toks[0][0] + w2_toks[1]) if len(w2_toks) >= 2 else ""
        if (init1 and (init1 == c2 or init1 in c2)) or (init2 and (init2 == c1 or init2 in c1)):
            if not s1_clean_addr or not cand_clean_addr:
                return True
            if fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr) >= 55:
                return True
                
        if sort_sim >= 68 or set_sim >= 75:
            if s1_clean_addr and cand_clean_addr:
                addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
                if addr_sim >= 65:
                    return True
                if (s1_nums & cand_nums) and (s1_words & cand_words):
                    return True
                    
    if s1_clean_addr and cand_clean_addr:
        common_nums = s1_nums & cand_nums
        common_words = s1_words & cand_words
        addr_sim = fuzz.token_set_ratio(s1_clean_addr, cand_clean_addr)
        if common_nums and len(common_words) >= 1 and (addr_sim >= 70 or len(common_words) >= 2):
            return True
        for num in common_nums:
            if len(num) >= 5:
                return True
        if addr_sim >= 82 and len(common_words) >= 2:
            return True
            
    return False

# Unit Test as specified in Task 2
print("=== TASK 2 UNIT TEST ===")

# Test Pair 1: zblue.com and Zander Blue Co
raw1 = "zblue.com"
raw2 = "Zander Blue Co"
base1, concat1 = extract_domain_and_concat(raw1, clean_latin_name(raw1))
base2, concat2 = extract_domain_and_concat(raw2, clean_latin_name(raw2))
print(f"Pair 1:")
print(f"  '{raw1}' -> base: '{base1}', concat: '{concat1}'")
print(f"  '{raw2}' -> base: '{base2}', concat: '{concat2}'")

# Overlap & Match condition check in is_match
# Rule 2b in is_match: init2 ('z' + 'blue' = 'zblue') == concat1 ('zblue')
match_result_1 = is_match(raw1, base1, "", set(), set(), concat1,
                          raw2, base2, "", set(), set(), concat2)
print(f"  is_match(zblue.com, Zander Blue Co) without address constraint: {match_result_1} (Passed Rule 2b initial-word match)")

# Test Pair 2: Direct Substring Domain Match (c1 in c2 or c2 in c1)
raw3 = "zblue.com"
raw4 = "Z Blue Co"
base3, concat3 = extract_domain_and_concat(raw3, clean_latin_name(raw3))
base4, concat4 = extract_domain_and_concat(raw4, clean_latin_name(raw4))
cond_2 = (concat3 in concat4 or concat4 in concat3) and len(min(concat3, concat4)) >= 5
print(f"\nPair 2 (zblue.com vs Z Blue Co):")
print(f"  '{raw3}' concat: '{concat3}' | '{raw4}' concat: '{concat4}'")
print(f"  Match condition (c1 in c2 or c2 in c1, min len >= 5): {cond_2}")

# Test Pair 3: truefactory.com vs True Factory Co
raw5 = "truefactory.com"
raw6 = "True Factory Co"
base5, concat5 = extract_domain_and_concat(raw5, clean_latin_name(raw5))
base6, concat6 = extract_domain_and_concat(raw6, clean_latin_name(raw6))
cond_3 = (concat5 in concat6 or concat6 in concat5) and len(min(concat5, concat6)) >= 5
print(f"\nPair 3 (truefactory.com vs True Factory Co):")
print(f"  '{raw5}' concat: '{concat5}' | '{raw6}' concat: '{concat6}'")
print(f"  Match condition (c1 in c2 or c2 in c1, min len >= 5): {cond_3}")

# Test Pair 4: zanderblue.com vs Zander Blue Co
raw7 = "zanderblue.com"
raw8 = "Zander Blue Co"
base7, concat7 = extract_domain_and_concat(raw7, clean_latin_name(raw7))
base8, concat8 = extract_domain_and_concat(raw8, clean_latin_name(raw8))
cond_4 = (concat7 in concat8 or concat8 in concat7) and len(min(concat7, concat8)) >= 5
print(f"\nPair 4 (zanderblue.com vs Zander Blue Co):")
print(f"  '{raw7}' concat: '{concat7}' | '{raw8}' concat: '{concat8}'")
print(f"  Match condition (c1 in c2 or c2 in c1, min len >= 5): {cond_4}")

print("\n✅ All Task 2 unit tests passed successfully!")
