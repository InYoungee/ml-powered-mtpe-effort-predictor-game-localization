import pandas as pd
import re
import difflib
import pickle

# ===== LOAD SAVED MODEL =====
with open('mt_effort_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print("✅ Model and feature columns loaded")

# ===== LOAD TEST DATA =====
df = pd.read_excel("mt_test_deepl.xlsx")
print(f"✅ Loaded {len(df)} test strings")
print(f"\nCategory distribution:")
print(df['Category_Consolidated'].value_counts())

# ===== CALCULATE SIMILARITY (DeepL vs Human) =====
def calc_similarity(str1, str2):
    """Calculate difflib similarity between two strings"""
    s1 = str(str1).strip().lower()
    s2 = str(str2).strip().lower()
    return difflib.SequenceMatcher(None, s1, s2).ratio()

print("\nCalculating DeepL vs Human similarity...")
df['Similarity_DeepL'] = df.apply(
    lambda row: calc_similarity(row['MT_DeepL'], row['EN_Confirmed_Trans']), axis=1
)

# ===== FEATURE EXTRACTION =====
def extract_features(row):
    """Extract same features used during training"""
    source = str(row['KO_Source'])
    category = row['Category_Consolidated']
    features = {}

    # Length features
    features['source_char_length'] = len(source)
    features['source_word_count'] = len(source.split())
    words = source.split()
    features['avg_char_per_word'] = len(source) / max(len(words), 1)

    # Special character features
    features['has_placeholder'] = int(bool(re.search(r'\{[0-9]+\}|\%[sd]|\$\{[\w]+\}', source)))
    features['placeholder_count'] = len(re.findall(r'\{[0-9]+\}|\%[sd]|\$\{[\w]+\}', source))
    features['has_html'] = int(bool(re.search(r'<[^>]+>', source)))
    features['html_tag_count'] = len(re.findall(r'<[^>]+>', source))
    features['has_numbers'] = int(bool(re.search(r'\d', source)))
    features['number_count'] = len(re.findall(r'\d+', source))
    features['has_quotes'] = int('"' in source or "'" in source)
    features['has_colon'] = int(':' in source)
    features['has_slash'] = int('/' in source)
    features['exclamation_count'] = source.count('!')
    features['question_count'] = source.count('?')

    # Language-specific features
    korean_chars = len(re.findall(r'[가-힣]', source))
    features['korean_char_ratio'] = korean_chars / max(len(source), 1)
    english_chars = len(re.findall(r'[a-zA-Z]', source))
    features['has_mixed_eng_ko'] = int(korean_chars > 0 and english_chars > 0)
    features['english_char_count'] = english_chars

    # Category features
    features['is_ui_system'] = int(category == 'UI_System')
    features['is_game_content'] = int(category == 'Game_Content')
    features['is_narrative'] = int(category == 'Narrative')
    features['is_marketing'] = int(category == 'Marketing')

    return features

print("Extracting features...")
feature_df = df.apply(extract_features, axis=1, result_type='expand')

# Ensure feature order matches training
X = feature_df[feature_columns]

# ===== PREDICT =====
print("Predicting effort labels...")
df['Predicted_Effort'] = model.predict(X)
df['Confidence'] = model.predict_proba(X).max(axis=1).round(3)

# ===== RESULTS SUMMARY =====
print("\n=== PREDICTION RESULTS ===")
print(f"\nPredicted effort distribution:")
print(df['Predicted_Effort'].value_counts())

print(f"\nPredictions by category:")
print(pd.crosstab(df['Category_Consolidated'], df['Predicted_Effort']))

print(f"\nAverage similarity by predicted effort:")
print(df.groupby('Predicted_Effort')['Similarity_DeepL'].mean().round(3))

# ===== SAVE RESULTS =====
output_cols = [
    'ID', 'Context', 'Category_Consolidated',
    'KO_Source', 'MT_DeepL', 'EN_Confirmed_Trans',
    'Similarity_DeepL', 'Predicted_Effort', 'Confidence'
]
df[output_cols].to_excel("prediction_results.xlsx", index=False)
print("\n✅ Results saved to: prediction_results.xlsx")

# ===== SAMPLE PREDICTIONS =====
print("\n=== SAMPLE PREDICTIONS (5 strings) ===")
sample = df[output_cols].head(5)
for _, row in sample.iterrows():
    print(f"\nID {row['ID']} [{row['Category_Consolidated']}]")
    print(f"  KO: {str(row['KO_Source'])[:60]}...")
    print(f"  DeepL similarity: {row['Similarity_DeepL']:.3f}")
    print(f"  Predicted effort: {row['Predicted_Effort'].upper()} (confidence: {row['Confidence']})")