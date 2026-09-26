import os
import csv
import re
import unicodedata
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz

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
    name = re.sub(r'(?<=[a-z])1\b', 'l', name)
    name = re.sub(r'[^\w\s]', ' ', name)
    return ' '.join(name.split())

ROAD_MAP = {
    r'\b(street|str)\b': 'st', r'\b(road)\b': 'rd', r'\b(avenue|ave\.)\b': 'ave',
    r'\b(boulevard|blvd\.)\b': 'blvd', r'\b(drive|dr\.)\b': 'dr', r'\b(lane|ln\.)\b': 'ln',
    r'\b(highway|hwy\.)\b': 'hwy', r'\b(rue|r\.)\b': 'rue', r'\b(allee|all\.)\b': 'allee'
}

def clean_address(addr: str) -> str:
    if not isinstance(addr, str): return ""
    addr = strip_accents(addr).lower()
    for pat, rep in ROAD_MAP.items():
        addr = re.sub(pat, rep, addr)
    addr = re.sub(r'\b(po box|pmb|hn|plot no|house no|flat no|unit|apt|suite)\b[\s\d-]*', ' ', addr)
    addr = re.sub(r'[^\w\s,]', ' ', addr)
    return ' '.join(addr.split())

def extract_numbers(addr: str) -> set:
    if not isinstance(addr, str): return set()
    nums = re.findall(r'\b\d+\b', addr)
    return {str(int(n)) for n in nums if len(n) <= 8}

def calc_f05(true_set, pred_set):
    if len(true_set) == 0 and len(pred_set) == 0: return 1.0
    if len(pred_set) == 0: return 0.0
    if len(true_set) == 0: return 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0: return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 1.25 * p * r / (0.25 * p + r)

print("Diagnostic script ready.")
