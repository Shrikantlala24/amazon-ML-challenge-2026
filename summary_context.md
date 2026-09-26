# Amazon ML Challenge 2026: Project Context & Discovery Summary

This document summarizes the actions performed during the initial context gathering phase, along with the detailed context, dataset profile, noise patterns, hardware constraints, and strategic recommendations for the **Business Entity Resolution Challenge**.

---

## 1. Executive Summary
- **Competition**: Amazon ML Challenge 2026
- **Problem Statement**: Business Entity Resolution (ER) across three heterogeneous, noisy data sources.
- **Anchor Reference Source**: **Source 1** (`S1-` IDs). For every entity in Source 1, find all corresponding matching records across **Source 2** (`S2-`) and **Source 3** (`S3-`).
- **Core Evaluation Metric**: Macro-averaged **$F_{0.5}$ Score** across all test Source 1 entities.
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - Heavily favors **Precision** (weighted 2× over Recall) to severely penalize false merges.
  - Macro-average treats singletons (no matches) as full credit (1.0) if predicted empty, and 0.0 if any false match is predicted.
- **🚨 CRITICAL COMPETITION UPDATE (Candidate Generation Scored for Final Ranking)**:
  > **"Candidate generation counts toward the final ranking. We will review your candidate_pairs.tsv and the code that produces it when deciding final rankings, alongside your matching_results.tsv score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard."**
  - **Strategic Impact**: We cannot use naive wide-net blocking (e.g. 50-100 candidates). Our blocking engine must be ultra-selective, aiming for an average of **$\approx 5 - 8$ candidates per entity** while preserving $>95\%$ recall.

---

## 2. Actions Performed During the Loop
During this session, the following exploratory and diagnostic actions were completed:

1. **Workspace & Resource Inspection**:
   - Inspected repository root directory, `.git`, `requirements.txt`, and `README.md`.
   - Explored `dataset_folder/student_resource/` which contains the official problem statement (`README.md`), documentation template (`Documentation_template.md`), submission validator script (`utils/validate_submission.py`), and raw dataset directories (`train/` and `test/`).
2. **Environment & Hardware Profiling**:
   - Checked Python runtime: **Python 3.11.4**.
   - Profiled available packages: `pandas (2.2.3)`, `xgboost (3.2.0)`, `scikit-learn (1.6.1)`, `torch (2.12.0 CPU)`, `transformers (5.8.1)`, `sentence-transformers (5.7.0)`, `pyarrow (22.0.0)`.
   - Checked Compute & Accelerators: **CUDA is False** (No local GPU available; CPU execution only).
   - Profiled System RAM and CPU: **12 logical CPU cores**, **7.68 GB Total RAM**, with **~0.45 GB available RAM** during testing.
3. **Data Profiling & Streaming Line Counting**:
   - Developed and executed a lightweight, low-memory streaming script (`scratch/quick_eda.py`) to process raw TSVs without loading complete files into memory.
   - Collected exact row counts, column schemas, country distributions, singleton percentages, and match cardinality distributions across all 7 dataset files (~24 million total rows).
4. **Noise Pattern Analysis on Ground Truth Samples**:
   - Created and executed `scratch/sample_matches.py` to extract ground-truth matched clusters and examine actual variations in business names and addresses.
5. **Empirical Verification of Country Partitioning**:
   - Created and executed `scratch/check_country_block.py` to check over 200,000 ground truth links for cross-country matches.
   - **Result**: Exactly **0 cross-country matches** found. Verified that `country` is an absolute hard partition.
6. **Test Set France Inspection**:
   - Created and executed `scratch/sample_france_test.py` to inspect French company records appearing only in the test set.
   - Categorized French-specific suffixes (`SARL`, `SASU`, `EURL`, `SCI`, `Association`) and street syntax (`Boulevard`, `Rue`, `Allée`, `bis`, etc.).
7. **Validator Script Analysis**:
   - Reviewed `utils/validate_submission.py` to analyze all formatting checks, required headers, ID checks, and candidate-pair subset validation.

---

## 3. Dataset Profiling & Distribution

### 3.1 File Overview

| Split | File Name | Records | File Size | Country Breakdown |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | **2,206,821** | 210.0 MB | US: 1,323,633 (60.0%), India: 883,188 (40.0%) |
| **Train** | `train_source2.tsv` | **5,034,616** | 489.3 MB | US: 3,016,817 (59.9%), India: 2,017,799 (40.1%) |
| **Train** | `train_source3.tsv` | **5,285,603** | 503.7 MB | US: 3,170,056 (60.0%), India: 2,115,547 (40.0%) |
| **Train GT**| `train_ground_truth.tsv` | **2,206,821** | 127.0 MB | Matches across S2 & S3 |
| **Test** | `test_source1.tsv` | **1,732,544** | 175.0 MB | India: 809,986 (46.8%), US: 663,106 (38.3%), **France: 259,452 (15.0%)** |
| **Test** | `test_source2.tsv` | **4,887,273** | 509.4 MB | India: 2,312,565 (47.3%), US: 1,871,330 (38.3%), **France: 703,378 (14.4%)** |
| **Test** | `test_source3.tsv` | **5,082,316** | 506.0 MB | India: 2,405,000 (47.3%), US: 1,945,701 (38.3%), **France: 731,615 (14.4%)** |
| **Total** | **7 Files** | **26,433,794** | **~2.52 GB** | Massive scale: 12.5M train records, 11.7M test records |

### 3.2 Ground Truth Matching Characteristics
- **Total S1 Reference Entities**: 2,206,821
- **Entities with Matches**: 2,083,574 (94.42%)
- **Singletons (No matches)**: 123,247 (5.58%)
- **Total Positive Links**:
  - Matched Source 2 Records: **3,693,619** (73.4% of S2 records are matches)
  - Matched Source 3 Records: **3,944,746** (74.6% of S3 records are matches)
  - Total Links: **7,638,365 pairs**
- **Match Count Distribution per Entity (Top occurrences)**:
  - 3 matches: 530,841 entities (24.1%)
  - 4 matches: 484,115 entities (21.9%)
  - 2 matches: 375,212 entities (17.0%)
  - 5 matches: 321,957 entities (14.6%)
  - 6 matches: 164,868 entities (7.5%)
  - 0 matches (singletons): 123,247 entities (5.58%)
  - 1 match: 119,157 entities (5.4%)
  - 7 matches: 63,968 entities (2.9%)
  - Mean matches per matched entity: **~3.66 matches**

---

## 4. Key Noise Patterns & Empirical Insights

### 4.1 Strict Country Partition
- We evaluated 200,004 ground-truth links and confirmed **0 cross-country matches**.
- **Implication**: Any candidate pair $(S_1, S_2)$ or $(S_1, S_3)$ where `country` does not match can be immediately pruned with 100% recall retention. This drastically reduces comparisons.

### 4.2 Open-Set Domain Shift: France in Test Set
- `France` appears **only in the test data** (~259,452 S1 records, ~1.43M S2/S3 records).
- Models and feature extractors must **not** be trained on one-hot country encodings or hardcoded US/India assumptions.
- Address parsers and tokenizers must handle French address components:
  - Legal forms: `SARL`, `SASU`, `SAS`, `EURL`, `SCI`, `Association`.
  - Street types: `Boulevard` (`BD`), `Rue` (`R.`), `Avenue` (`AV`), `Allée`, `Impasse`, `Chemin`, `bis`.
  - Diacritics & Accents: `é`, `è`, `à`, `ç`, `ê`, `î`, `ô`.

### 4.3 Observed Variations in Names
1. **Legal Suffix Inconsistencies**: `Inc` vs `Incorporated`, `Pvt Ltd` vs `Private Limited`, `LLC`, `Corp`, `Services`.
2. **Affixes & DBA (Doing Business As)**: e.g. `Ectosyn dba Christ Chapel` matching `Christ Chapel`.
3. **Leetspeak / Typographical / OCR Noise**: e.g. `Christ [Chape1]` where `1` replaces `l`.
4. **Social & Web Noise**: `@handle`, `.com`, bracketed IDs like `(ID: 84923)`.
5. **Punctuation & Word Order**: `Callicoat & Dailey Inc` vs `Dailey and Callicoat`.

### 4.4 Observed Variations in Addresses
1. **Component Inversion / Permutation**:
   - S1: `OH, Columbus, 5559 Orville Avenue`
   - S2: `555 ORVILLE AVENUE, PO BOX 3877, COLUMBUS, OH`
2. **State & Street Abbreviations**:
   - `NC` vs `North Carolina`, `OK` vs `Oklahoma`, `AZ` vs `Arizona`.
   - `Dr` vs `Drive`, `Rd` vs `Road`, `Ave` vs `Avenue`.
3. **Number Variations**:
   - Leading zeroes: `17560` vs `0017560`.
   - Building/Plot qualifiers: `#1795`, `Plot 334 797`, `HN 567 797`.
4. **Multilingual / Transliteration**:
   - English vs vernacular scripts (e.g. `West Bengal` vs Bengali `পশ্চিমবঙ্গ`).
5. **Missing Address Fields**:
   - Certain S2/S3 records have completely empty address fields (`""`). Name matching alone must resolve these cases.

---

## 5. Hardware Constraints & Engineering Architecture

### 5.1 System Limits
- **CPU**: 12 logical cores (10 physical cores).
- **RAM**: 7.68 GB total, with **~0.45 GB available RAM**.
- **GPU**: None (CPU-only).

### 5.2 Critical Engineering Guardrails
1. **Never load full dataset into memory at once**: A single pandas DataFrame with 5.2M text rows takes >1.5 GB in memory, which will trigger immediate OS Out-Of-Memory (OOM) termination on this system.
2. **Chunking / Streaming / Disk Storage**:
   - Use streaming TSV parsers (e.g., Python `csv` reader, Polars scan / streaming mode, PyArrow memory mapping, or DuckDB/SQLite for indexing).
3. **Country-Partitioned Processing**:
   - Process US, India, and France completely separately in memory chunks to divide dataset size by 3.
4. **Two-Stage Pipeline (Standard Entity Resolution Paradigm)**:
   - **Stage 1: Blocking / Candidate Generation**:
     - Fast candidate retrieval reducing $1.7\text{M} \times 9.9\text{M}$ test space down to $\leq 15-20$ candidates per S1.
     - Multi-key blocking: Standardized Name Prefix/Tokens + Street Numbers + City/State Tokens + MinHash / character n-grams.
   - **Stage 2: Pairwise Feature Engineering & Classification**:
     - String distance metrics (RapidFuzz, Jaccard, Token Sort, Levenshtein, Dice, Longest Common Subsequence).
     - Fast Gradient Boosted Trees (LightGBM / XGBoost) trained to optimize the **$F_{0.5}$ metric**.
     - High precision thresholding to safeguard against false positives.

---

## 6. Official Submission & Compliance Rules

### 6.1 Output Format
Two TSV files placed in `output/`:
1. **`matching_results.tsv`** (Only file scored on the leaderboard):
   ```tsv
   source1_entity_id	matched_entity_ids
   S1-00001	S2-00047,S2-00193,S3-00812
   S1-00002	S3-00004
   S1-00003	
   ```
   - Must contain every single test S1 entity ID (exact 1-to-1 match with `test_source1.tsv`).
   - Singletons must have an empty `matched_entity_ids` field.
   - Comma-separated IDs, no duplicate IDs, no self-matches (`S1-` IDs forbidden).
2. **`candidate_pairs.tsv`** (Candidate set from blocking stage):
   - Every matched ID in `matching_results.tsv` must be a subset of `candidate_pairs.tsv`.
   - Shows candidate generation coverage and reduction ratio.

### 6.2 Strict Restrictions
- **Zero External Data / Lookup APIs**: Strictly prohibited from using external databases, geocoders, search engines, or commercial ER APIs.
- **Model Constraints**: MIT or Apache 2.0 open-source models up to 8 Billion parameters.
- **Final Zip Packaging**:
  ```
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
  ```

---

## 7. Next Recommended Steps
1. **Local Validation Scheme**: Construct a stratified 100k or 200k validation split from `train/` mirroring the test distribution (including singletons) and write an exact macro $F_{0.5}$ scorer.
2. **Text Normalization Module**: Build country-agnostic and country-specific text cleaning routines (cleaning accents, normalizing legal suffixes across EN and FR, parsing street numbers).
3. **Scalable Blocking Engine**: Implement high-recall, high-speed multi-index blocking capable of running efficiently within the 8 GB RAM boundary.
4. **Classifier & Scoring**: Engineer similarity features and calibrate an XGBoost / LightGBM decision threshold targeting the precision-skewed $F_{0.5}$ objective.
