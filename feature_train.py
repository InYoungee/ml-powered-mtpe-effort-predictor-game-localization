import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ===== LOAD DATA =====
df = pd.read_excel("strings_with_analysis.xlsx")
print(f"Loaded {len(df)} strings")
print(f"\nEffort_Label distribution:")
print(df['Effort_Label'].value_counts())

# ===== LABEL SANITY CHECK =====
# Verify labels make logical sense before training
print("\n=== LABEL SANITY CHECK ===")
print("Average Best_MT_Similarity by Effort_Label:")
print(df.groupby('Effort_Label')['Best_MT_Similarity'].mean().round(3))
# Expected order: light > moderate > heavy
# If this holds, your labels are meaningful

# ===== FEATURE EXTRACTION =====
def extract_features(row):
    """Extract features from a KO source string that predict MT post-edit effort"""
    source = str(row['KO_Source'])
    category = row['Category_Consolidated']
    features = {}

    # ===== LENGTH FEATURES =====
    features['source_char_length'] = len(source)
    features['source_word_count'] = len(source.split())
    words = source.split()
    features['avg_char_per_word'] = len(source) / max(len(words), 1)

    # ===== SPECIAL CHARACTER FEATURES =====
    # Placeholders like {0}, %s, ${var}
    features['has_placeholder'] = int(bool(re.search(r'\{[0-9]+\}|\%[sd]|\$\{[\w]+\}', source)))
    features['placeholder_count'] = len(re.findall(r'\{[0-9]+\}|\%[sd]|\$\{[\w]+\}', source))

    # HTML tags
    features['has_html'] = int(bool(re.search(r'<[^>]+>', source)))
    features['html_tag_count'] = len(re.findall(r'<[^>]+>', source))

    # Numbers
    features['has_numbers'] = int(bool(re.search(r'\d', source)))
    features['number_count'] = len(re.findall(r'\d+', source))

    # Special punctuation
    features['has_quotes'] = int('"' in source or "'" in source)
    features['has_colon'] = int(':' in source)
    features['has_slash'] = int('/' in source)
    features['exclamation_count'] = source.count('!')
    features['question_count'] = source.count('?')

    # ===== LANGUAGE-SPECIFIC FEATURES =====
    # Korean characters (Hangul)
    korean_chars = len(re.findall(r'[가-힣]', source))
    features['korean_char_ratio'] = korean_chars / max(len(source), 1)

    # English characters mixed in Korean (like "All속성")
    english_chars = len(re.findall(r'[a-zA-Z]', source))
    features['has_mixed_eng_ko'] = int(korean_chars > 0 and english_chars > 0)
    features['english_char_count'] = english_chars

    # ===== CATEGORY FEATURES (One-Hot Encoding) =====
    features['is_ui_system'] = int(category == 'UI_System')
    features['is_game_content'] = int(category == 'Game_Content')
    features['is_narrative'] = int(category == 'Narrative')
    features['is_marketing'] = int(category == 'Marketing')

    return features

print("\nExtracting features...")
feature_df = df.apply(extract_features, axis=1, result_type='expand')
feature_columns = feature_df.columns.tolist()

print(f"✅ Extracted {len(feature_columns)} features")
print(f"Feature names: {feature_columns}")

# Save feature file for reference (optional)
df_with_features = pd.concat([df, feature_df], axis=1)
df_with_features.to_excel("strings_with_features.xlsx", index=False)
print("✅ Features saved to: strings_with_features.xlsx")

# ===== PREPARE TRAINING DATA =====
# Option A: Source features only (predict effort BEFORE seeing MT output)
X_source_only = feature_df

# Option B: Source features + MT similarity scores (predict effort AFTER MT runs)
similarity_columns = ['Similarity_Google', 'Similarity_DeepL', 'Best_MT_Similarity']
X_with_similarity = pd.concat([feature_df, df[similarity_columns]], axis=1)

y = df['Effort_Label']

print(f"\n=== TRAINING DATA ===")
print(f"Total samples: {len(y)}")
print(f"Source-only features: {len(feature_columns)}")
print(f"Features with similarity scores: {len(feature_columns) + len(similarity_columns)}")

# ===== TRAIN/TEST SPLIT =====
# Using source-only features as primary model
X_train, X_test, y_train, y_test = train_test_split(
    X_source_only, y,
    test_size=0.2,       # 20% held out for testing
    random_state=42,     # reproducible results
    stratify=y           # keeps label balance in both sets
)

print(f"\nTraining set: {len(X_train)} strings")
print(f"Test set:     {len(X_test)} strings")

# ===== MODEL A: Random Forest =====
print("\n=== TRAINING: Random Forest ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'   # handles imbalanced classes
)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)
y_pred_rf = rf_model.predict(X_test)

print(f"Accuracy: {rf_acc * 100:.1f}%")
print(classification_report(y_test, y_pred_rf))

# ===== MODEL B: Gradient Boosting =====
print("\n=== TRAINING: Gradient Boosting ===")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_acc = gb_model.score(X_test, y_test)
y_pred_gb = gb_model.predict(X_test)

print(f"Accuracy: {gb_acc * 100:.1f}%")
print(classification_report(y_test, y_pred_gb))

# ===== PICK BEST MODEL =====
best_model = rf_model if rf_acc >= gb_acc else gb_model
best_pred = y_pred_rf if rf_acc >= gb_acc else y_pred_gb
best_name = "Random Forest" if rf_acc >= gb_acc else "Gradient Boosting"

print(f"\n=== BEST MODEL: {best_name} ({max(rf_acc, gb_acc)*100:.1f}%) ===")

# ===== CONFUSION MATRIX =====
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred, labels=['light', 'moderate', 'heavy'])
cm_df = pd.DataFrame(
    cm,
    index=['Actual: light', 'Actual: moderate', 'Actual: heavy'],
    columns=['Predicted: light', 'Predicted: moderate', 'Predicted: heavy']
)
print(cm_df)

# ===== FEATURE IMPORTANCE =====
print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# ===== BONUS: Compare with similarity scores =====
print("\n=== BONUS: WITH vs WITHOUT similarity scores ===")
X_train_sim, X_test_sim, y_train_sim, y_test_sim = train_test_split(
    X_with_similarity, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
rf_sim = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
)
rf_sim.fit(X_train_sim, y_train_sim)
sim_acc = rf_sim.score(X_test_sim, y_test_sim)

print(f"Source features only:          {max(rf_acc, gb_acc)*100:.1f}%")
print(f"Source features + MT similarity: {sim_acc*100:.1f}%")
print("(Gap shows how much the MT output quality itself drives effort prediction)")

# ===== SAVE MODEL =====
with open('mt_effort_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print(f"\n✅ Best model ({best_name}) saved to: mt_effort_model.pkl")
print("✅ Feature columns saved to: feature_columns.pkl")