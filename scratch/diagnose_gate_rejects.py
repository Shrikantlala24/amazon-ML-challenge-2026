import csv
import re
import unicodedata
from rapidfuzz import fuzz

def clean_latin_name(name: str) -> str:
    if not name or not isinstance(name, str): return ""
    text = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII').lower()
    text = re.sub(r'\[\[.*?\]\]|\[.*?\]|\(.*?\)', ' ', text)
    text = re.sub(r'\b(d\.?b\.?a|trading as|t/a)\b.*', ' ', text)
    text = re.sub(r'\b(corp|corporation|inc|incorporated|llc|l\.l\.c|ltd|limited|co|company|services|partners|private|pvt|sarl|sas|sasu|eurl|sci)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

TLD_REGEX = re.compile(r'\.(com|org|net|in|co|io|biz|info|fr|gov|edu)\b.*$', flags=re.IGNORECASE)

def extract_domain_and_concat(raw_name: str, clean_name: str = None):
    if not raw_name or not isinstance(raw_name, str): return "", ""
    if '.' in raw_name:
        stripped = TLD_REGEX.sub('', raw_name)
        if stripped != raw_name:
            base = clean_latin_name(stripped)
        else:
            base = clean_name if clean_name is not None else clean_latin_name(raw_name)
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

def is_match_params(s1_name, s1_addr, c_name, c_addr,
                    t_sort_high=84, t_set_high=88, t_sort_med=68, t_set_med=75, t_addr_med=65):
    n1 = clean_latin_name(s1_name)
    n2 = clean_latin_name(c_name)
    a1 = clean_address(s1_addr)
    a2 = clean_address(c_addr)
    nums1, words1 = extract_addr_tokens(s1_addr)
    nums2, words2 = extract_addr_tokens(c_addr)
    _, concat1 = extract_domain_and_concat(s1_name, n1)
    _, concat2 = extract_domain_and_concat(c_name, n2)

    sort_sim = fuzz.token_sort_ratio(n1, n2) if (n1 and n2) else 0
    set_sim = fuzz.token_set_ratio(n1, n2) if (n1 and n2) else 0
    addr_sim = fuzz.token_set_ratio(a1, a2) if (a1 and a2) else 0

    matched = False
    rule = ''

    if n1 and n2:
        if n1 == n2:
            matched, rule = True, 'exact_name'
        elif sort_sim >= t_sort_high:
            matched, rule = True, f'sort_high>={t_sort_high}'
        elif set_sim >= t_set_high:
            matched, rule = True, f'set_high>={t_set_high}'
        elif concat1 and concat2 and (concat1 in concat2 or concat2 in concat1) and len(min(concat1, concat2)) >= 5:
            if not a1 or not a2 or addr_sim >= 55:
                matched, rule = True, 'domain_substring'
        else:
            w1_toks = n1.split()
            w2_toks = n2.split()
            init1 = (w1_toks[0][0] + w1_toks[1]) if len(w1_toks) >= 2 else ''
            init2 = (w2_toks[0][0] + w2_toks[1]) if len(w2_toks) >= 2 else ''
            if (init1 and (init1 == concat2 or init1 in concat2)) or (init2 and (init2 == concat1 or init2 in concat1)):
                if not a1 or not a2 or addr_sim >= 55:
                    matched, rule = True, 'initial_word'
            elif (sort_sim >= t_sort_med or set_sim >= t_set_med):
                if a1 and a2:
                    if addr_sim >= t_addr_med:
                        matched, rule = True, f'med_name+addr>={t_addr_med}'
                    elif (nums1 & nums2) and (words1 & words2):
                        matched, rule = True, 'med_name+nums&words'

    if not matched and a1 and a2:
        common_nums = nums1 & nums2
        common_words = words1 & words2
        if common_nums and len(common_words) >= 1 and (addr_sim >= 70 or len(common_words) >= 2):
            matched, rule = True, 'addr_unit+words'
        elif any(len(num) >= 5 for num in common_nums):
            matched, rule = True, 'pin_code'
        elif addr_sim >= 82 and len(common_words) >= 2:
            matched, rule = True, 'addr_high>=82'

    return matched, rule, sort_sim, set_sim, addr_sim

# Load 300 ground truth pairs
gt_pairs = []
with open('dataset_folder/student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        s1_id = row[0]
        if len(row) >= 2 and row[1].strip():
            targets = row[1].split(',')
            for t in targets:
                gt_pairs.append((s1_id, t))
        if len(gt_pairs) >= 300: break

s1_needed = {p[0] for p in gt_pairs}
target_needed = {p[1] for p in gt_pairs}

s1_data = {}
with open('dataset_folder/student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        if row[0] in s1_needed:
            s1_data[row[0]] = (row[1], row[2], row[3])
            if len(s1_data) >= len(s1_needed): break

target_data = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(f'dataset_folder/student_resource/dataset/train/{fn}', 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        next(r)
        for row in r:
            if row[0] in target_needed:
                target_data[row[0]] = (row[1], row[2], row[3])

# Diagnose rejects
rejects = []
for s1_id, t_id in gt_pairs:
    if s1_id in s1_data and t_id in target_data:
        s1_n, s1_a, country = s1_data[s1_id]
        t_n, t_a, _ = target_data[t_id]
        m, rule, sort_sim, set_sim, addr_sim = is_match_params(s1_n, s1_a, t_n, t_a)
        if not m:
            rejects.append({
                's1_id': s1_id, 't_id': t_id,
                's1_name': s1_n, 't_name': t_n,
                's1_addr': s1_a, 't_addr': t_a,
                'sort_sim': sort_sim, 'set_sim': set_sim, 'addr_sim': addr_sim
            })

print(f'Total true pairs checked: {len(gt_pairs)}')
print(f'Total matcher gate rejects: {len(rejects)} ({len(rejects)/len(gt_pairs):.1%})')
print(f'\nTop {min(35, len(rejects))} Matcher Gate Rejects and their similarity values:')
for i, rej in enumerate(rejects[:35], 1):
    print(f"{i:2d}. {rej['s1_id']} to {rej['t_id']} | sort_sim={rej['sort_sim']:2.0f} set_sim={rej['set_sim']:2.0f} addr_sim={rej['addr_sim']:2.0f} | n1='{rej['s1_name'][:25]}' n2='{rej['t_name'][:25]}'")
