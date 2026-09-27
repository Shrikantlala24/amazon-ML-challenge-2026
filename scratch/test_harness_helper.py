import sys, os, time, re, csv, unicodedata
from collections import defaultdict, Counter
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

def clean_vernacular_name(name: str) -> str:
    if not name or not isinstance(name, str): return ""
    text = name.lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def extract_domain_and_concat(raw_name: str, clean_name: str = None):
    if not raw_name or not isinstance(raw_name, str):
        return "", ""
    stripped = re.sub(r'\.(com|org|net|in|co|io|biz|info|fr|gov|edu)\b.*$', '', raw_name, flags=re.IGNORECASE)
    base = clean_latin_name(stripped)
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

def make_keys(raw_name, raw_addr, country):
    keys = set()
    c_name = clean_latin_name(raw_name)
    v_name = clean_vernacular_name(raw_name)
    base_dom, concat = extract_domain_and_concat(raw_name, c_name)
    nums, words = extract_addr_tokens(raw_addr)
    
    if len(concat) >= 4:
        keys.add(f"cc:{country}:{concat[:8]}")
        keys.add(f"cc5:{country}:{concat[:5]}")
        
    w_list = c_name.split()
    if w_list:
        if len(w_list[0]) >= 3:
            keys.add(f"fw:{country}:{w_list[0]}")
        if len(w_list) >= 2:
            keys.add(f"w2:{country}:{w_list[0]}_{w_list[1]}")
            keys.add(f"init:{country}:{w_list[0][0]}{w_list[1]}")
        elif len(w_list) == 1:
            keys.add(f"init:{country}:{w_list[0]}")
            
        if len(c_name) <= 22:
            keys.add(f"cn:{country}:{c_name}")
            
        for w in w_list:
            if len(w) >= 6:
                keys.add(f"ng3:{country}:{w[:3]}_{w[-3:]}")
    elif v_name:
        v_tokens = v_name.split()
        if v_tokens:
            keys.add(f"vn:{country}:{v_tokens[0]}")
            if len(v_tokens) >= 2:
                keys.add(f"vn2:{country}:{v_tokens[0]}_{v_tokens[1]}")
                
    w_list_sorted = sorted(list(words))
    for num in nums:
        for w in w_list_sorted[:3]:
            keys.add(f"an:{country}:{num}_{w}")
        if len(num) >= 5:
            keys.add(f"pin:{country}:{num}")
            
    for i in range(min(3, len(w_list_sorted))):
        for j in range(i + 1, min(4, len(w_list_sorted))):
            keys.add(f"aw2:{country}:{w_list_sorted[i]}_{w_list_sorted[j]}")
            
    return keys

print("Test setup complete.")
