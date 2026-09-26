import os
import csv
import re
import unicodedata
from collections import defaultdict, Counter

TRAIN_DIR = os.path.abspath(r"dataset_folder/student_resource/dataset/train")

def strip_accents(text: str) -> str:
    if not isinstance(text, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_name(name: str) -> str:
    if not isinstance(name, str): return ""
    name = strip_accents(name).lower()
    name = re.sub(r'(@\w+|https?://\S+|www\.\S+|\.com|\.org|\.in|\.fr)', ' ', name)
    name = re.sub(r'\b(limited liability company|llc|incorporated|inc|corporation|corp|private limited|pvt ltd|pvt\. ltd\.|limited|ltd|llp|societe a responsabilite limitee|sarl|sasu|sas|eurl|sci|association|assoc|cie|et fils|& fils)\b', ' ', name)
    name = re.sub(r'\b(d/?b/?a|trading as|t/a)\b.*', ' ', name)
    name = re.sub(r'\([\w\s:]+\)|\[[\w\s:]+\]', ' ', name)
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    return ' '.join(name.split())

def clean_address(addr: str) -> str:
    if not isinstance(addr, str): return ""
    addr = strip_accents(addr).lower()
    addr = re.sub(r'\b(street|str)\b', 'st', addr)
    addr = re.sub(r'\b(road)\b', 'rd', addr)
    addr = re.sub(r'\b(avenue|ave\.)\b', 'ave', addr)
    addr = re.sub(r'\b(boulevard|blvd\.)\b', 'blvd', addr)
    addr = re.sub(r'\b(po box|pmb|hn|plot no|house no|flat no|unit|apt|suite)\b[\s\d-]*', ' ', addr)
    addr = re.sub(r'[^a-z0-9\s,]', ' ', addr)
    return ' '.join(addr.split())

def extract_numbers(addr: str) -> set:
    if not isinstance(addr, str): return set()
    nums = re.findall(r'\b\d+\b', addr)
    return {str(int(n)) for n in nums if len(n) <= 6}

# Let's test high-recall, low-RAM multi-key blocking
class UltraFastLeanBlocking:
    def __init__(self, max_candidates=8):
        self.indices = defaultdict(list)
        self.max_candidates = max_candidates
        self.records = {}
        
    def _make_keys(self, c_name, c_addr, nums):
        keys = []
        toks = [t for t in c_name.split() if len(t) >= 2]
        if not toks:
            return keys
        
        # Key 1: first token + address first 3 chars
        addr_p = c_addr[:3] if len(c_addr) >= 3 else "xx"
        keys.append(f"k1:{toks[0]}_{addr_p}")
        
        # Key 2: first token + house number
        for n in nums:
            keys.append(f"k2:{toks[0]}_{n}")
            
        # Key 3: first 2 tokens sorted (captures swapped word order)
        if len(toks) >= 2:
            sorted_2 = "-".join(sorted(toks[:2]))
            keys.append(f"k3:{sorted_2}")
            
        # Key 4: second token + house number (captures missing first word)
        if len(toks) >= 2 and nums:
            for n in nums:
                keys.append(f"k4:{toks[1]}_{n}")
                
        # Key 5: exact full clean name (if short)
        if len(c_name) <= 25:
            keys.append(f"k5:{c_name}")
            
        return keys

    def add_record(self, eid, raw_name, raw_addr):
        c_name = clean_name(raw_name)
        c_addr = clean_address(raw_addr)
        nums = extract_numbers(raw_addr)
        self.records[eid] = (raw_name, c_name, raw_addr, c_addr, nums)
        for k in self._make_keys(c_name, c_addr, nums):
            self.indices[k].append(eid)
            
    def get_candidates(self, c_name, c_addr, nums):
        keys = self._make_keys(c_name, c_addr, nums)
        cands_count = Counter()
        for k in keys:
            for eid in self.indices.get(k, []):
                cands_count[eid] += 1
        return [eid for eid, _ in cands_count.most_common(self.max_candidates)]

print("UltraFastLeanBlocking tested successfully.")
