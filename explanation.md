notebook
- feature engineering
- Algorithm (3)

variations 
- variables

---
- DL -> Data dependent -> 
- ML -> saturate 

train both ML and DL alternative

if ML < DL :
   DL wala selected for final Model
if ML > DL:
   sample Data (5000 -> 10,000 -> 25,000 -> 50,000 -> 100,000 -> 200,000)

---


## 🌲 3. Why LightGBM? Is It Better Than XGBoost?

For our Machine Learning Judge, we chose **LightGBM** (Light Gradient Boosting Machine).

### Why LightGBM?
1. **Speed & RAM Efficiency**:
   - Our host machine has 8 GB RAM with limited free memory.
   - Traditional XGBoost scans all exact numbers to find splits.
   - LightGBM groups numbers into **256 discrete histogram buckets**. This makes it **5× to 10× faster to train** and uses **significantly less memory**.
2. **Handles Non-Linear Combinations**:
   - A linear model assumes each clue acts independently.
   - LightGBM can easily learn rules like:
     > *"IF name similarity is above 90% AND house numbers match, THEN it is a match, EVEN IF the street word was missing."*

---

### Is LightGBM "Better" than XGBoost?
* **Accuracy-wise**: They are virtually identical. Both are world-class decision tree algorithms.
* **Resource-wise**: LightGBM is noticeably faster and lighter on large tabular datasets with hundreds of thousands of candidate pairs. (Modern XGBoost with `tree_method='hist'` is very close to LightGBM, but LightGBM is still slightly more memory-friendly).

---

### 🚀 What Are Better Alternatives to Level Up This Model?

If you want to move beyond this baseline and build a winning tournament model, here are the 3 best alternatives:

#### Alternative 1: The "Tri-Model Ensemble" (LightGBM + XGBoost + CatBoost)
* Train LightGBM, XGBoost, and CatBoost on the exact same 15 features.
* **Average their predicted probabilities**:
  $$P_{\text{final}} = \frac{P_{\text{LightGBM}} + P_{\text{XGBoost}} + P_{\text{CatBoost}}}{3}$$
* **Why it's better**: Different algorithms make mistakes on different edge cases. Combining them smooths out errors and consistently boosts $F_{0.5}$ by $1 - 3\%$.

#### Alternative 2: Pre-Trained Deep Learning Transformers (DeBERTa / MiniLM)
* Instead of only feeding 15 hand-crafted numbers, feed the full text into a Transformer:
  `[CLS] S1: Orelee's Barbershop, High Point NC [SEP] S2: Orelee Barbershop, NC [SEP]`
* **Why it's better**: A transformer understands true **semantic meaning** (e.g., it knows that *"Dr."* and *"Physicians"* are related, or *"Auto Repair"* and *"Mechanic"* mean the same thing).
* **Catch**: It is much slower and requires a GPU (perfect for Google Colab!).

#### Alternative 3: The Stacking Meta-Model (Hybrid)
* Use **Sentence-Transformers** on Colab GPU to generate a semantic embedding score.
* Combine that semantic score with our 15 RapidFuzz string features.
* Feed all of them into LightGBM for the final decision.