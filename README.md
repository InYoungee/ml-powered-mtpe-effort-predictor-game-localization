# ML-Powered MT Post-Edit Effort Predictor for Game Localization

A machine learning project that predicts post-edit effort for Korean-to-English machine translation in game localization. Built by a localization professional with 6+ years of localization project management experience, combining domain expertise with Python-based ML to automate quality estimation at scale as part of an ongoing exploration of AI/ML applications in localization workflows.

---

## Background & Motivation

In game localization, machine translation (MT) is increasingly used to accelerate delivery. However, MT output quality varies widely — some strings require minimal editing while others need near-complete rewrites. Identifying which strings need heavy post-editing *before* the editing begins helps localization managers:

- Prioritize workload and allocate linguist resources more effectively
- Estimate project costs and timelines more accurately
- Evaluate MT engine performance by content type

Traditional QE (Quality Estimation) tools exist but are expensive, require external API integration, and are not tailored to game-specific content (HTML tags, placeholders, mixed KO/EN strings, game terminology). This project builds a lightweight, domain-specific QE model trained on real KO-EN game localization data.

---

## Dataset

- **819 KO-EN string pairs** collected from **7 different published games** (2025)
- Content categories: Game Content, UI/System, Marketing, Narrative
- Each string includes: KO source, MT output (Google Translate & DeepL), and human-confirmed English translation
- Dataset spans multiple game titles and content types, improving generalizability beyond a single game's style or terminology
- **Data not included in this repository** due to IP considerations — the app accepts user-uploaded files in the same format

---

## Methodology

### 1. Similarity Scoring
Post-edit effort labels were derived by measuring similarity between MT output and human translation using Python's `difflib.SequenceMatcher`:

```python
similarity = difflib.SequenceMatcher(None, mt_output, human_translation).ratio()
```

For each string, the better-performing MT engine (Google or DeepL) was selected, and its similarity score against the human translation was used for labeling.

### 2. Effort Labeling
Similarity scores were converted into three effort categories:

| Label | Similarity Threshold | Meaning |
|---|---|---|
| **Light** | ≥ 0.85 | MT is close to human translation, minor edits needed |
| **Moderate** | ≥ 0.60 | MT needs meaningful editing |
| **Heavy** | < 0.60 | MT output requires significant rewriting |

**Label distribution:**
- Moderate: 295 strings (36%)
- Heavy: 289 strings (35%)
- Light: 235 strings (29%)

### 3. Feature Engineering
21 features were extracted from KO source strings:

**Length features:** source character length, word count, average characters per word

**Special character features:** placeholder count, HTML tag count, number count, punctuation patterns (quotes, colons, slashes, exclamation marks, question marks)

**Language-specific features:** Korean character ratio, mixed KO/EN detection, English character count

**Content category features:** one-hot encoded (UI_System, Game_Content, Narrative, Marketing)

### 4. Model Training
Two classifiers were trained and compared:

```python
RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
```

- 80/20 train/test split with stratification
- Best model selected automatically and saved via pickle

---

## Key Findings

### Model Performance
- **Gradient Boosting outperformed Random Forest**: 59.8% vs 57.9% accuracy
- Both significantly above random baseline (~33% for 3-class classification)
- "Moderate" was the hardest class to predict — inherently ambiguous between light and heavy

### Most Important Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | korean_char_ratio | 27.1% |
| 2 | source_char_length | 19.2% |
| 3 | avg_char_per_word | 15.7% |
| 4 | source_word_count | 8.9% |
| 5 | is_narrative | 4.9% |

Character-level and length features dominate — longer, more complex strings with lower Korean character ratios (mixed KO/EN content) tend to require heavier post-editing.

### Source Features vs. MT Similarity Scores

| Feature Set | Accuracy |
|---|---|
| Source string features only | 59.8% |
| Source features + MT similarity scores | 98.8% |

This gap reveals that **source string characteristics alone are a weak predictor of MT quality** — the MT output itself carries most of the quality signal. This is an important finding: predicting effort *before* MT runs is fundamentally harder than predicting it *after*.

### MT Engine Comparison: Google vs. DeepL
- Google Translate performed better on 343 strings
- DeepL performed better on 274 strings
- 202 strings were tied
- Neither engine consistently outperformed the other across all content categories

---

## difflib vs. Levenshtein: A Methodological Comparison

Two string similarity metrics were evaluated for generating effort labels:

**difflib (SequenceMatcher):**
- Finds longest common subsequences — measures structural/phrase-level similarity
- More forgiving of word-order differences
- Average score: ~0.64 (Google), ~0.62 (DeepL)

**Levenshtein distance:**
- Counts minimum character-level edits (insert, delete, substitute)
- Stricter — penalizes every character difference
- Average score: ~0.47 (Google), ~0.48 (DeepL)
- Industry standard in TMS tools (memoQ, Trados fuzzy match scoring)

**Key finding: 82.8% of strings received different effort labels depending on which metric was used.** Levenshtein's stricter scoring pushed labels heavily toward "heavy" (463 vs. 289 with difflib), collapsing "light" strings from 235 to just 74.

This highlights a critical methodological consideration: **effort label thresholds must be calibrated per metric** — thresholds valid for difflib cannot be reused with Levenshtein scores.

This project uses difflib as the primary metric. Levenshtein-based retraining with recalibrated thresholds is identified as a future improvement.

---

## Limitations

**Label quality:** Effort labels are derived from MT-to-human similarity scores, not from actual time-tracked post-editing effort. Text similarity and editing effort are correlated but not identical — a short string with a wrong term may score low similarity but take seconds to fix, while a fluent but subtly mistranslated string may score high but require significant cognitive effort.

**Dataset size:** 819 strings across 3 classes (~270 per class) is relatively small for robust ML classification. Performance would likely improve with more data.

**Domain specificity:** Although the model was trained across 7 game titles, all data comes from KO-EN game localization. Performance on other language pairs, genres, or non-game content may vary.

**Source-only prediction ceiling:** As shown in the findings, source features alone achieve ~60% accuracy. Without MT output, reliable effort prediction is inherently limited.

---

## Future Improvements

- Retrain using **actual post-editing time** tracked in a TMS (e.g. memoQ) as labels
- Evaluate **Levenshtein-based labels** with recalibrated thresholds
- Add **semantic similarity features** using sentence transformers (e.g. `paraphrase-multilingual-MiniLM`)
- Expand to **binary classification** (light vs. heavy) for higher accuracy
- Test on **non-game content** (marketing, UI, documentation) to evaluate generalizability
- Build **Streamlit web app** for interactive prediction (in progress)

---

## Project Structure

```
├── feature_train.py          # Feature extraction + model training
├── predict_new_strings.py    # Prediction script for new strings
├── app.py                    # Streamlit web app
├── mt_effort_model.pkl       # Trained Gradient Boosting model
├── feature_columns.pkl       # Feature column names for prediction
├── requirements.txt          # Python dependencies
└── README.md
```

---

## How to Run

### Requirements
```bash
pip install pandas scikit-learn openpyxl
```

### Training
```bash
python feature_train.py
```
Expects `strings_with_analysis.xlsx` with columns: `KO_Source`, `MT_Google`, `MT_DeepL`, `EN_Confirmed_Trans`, `Category_Consolidated`, `Effort_Label`

### Prediction on New Strings
```bash
python predict_new_strings.py
```
Expects `mt_test_deepl.xlsx` with columns: `KO_Source`, `MT_DeepL`, `EN_Confirmed_Trans`, `Category_Consolidated`

---

## Tech Stack

- **Python** — pandas, scikit-learn, difflib, re, pickle
- **Models** — RandomForestClassifier, GradientBoostingClassifier
- **App** — Streamlit, Plotly
- **Data** — Real KO-EN game localization strings (819 pairs, 2025)

---
## Demo
![ml_mtpedemo](https://github.com/InYoungee/ml-powered-mtpe-effort-predictor-game-localization/blob/main/images/ml_mtpe_demo.gif)

