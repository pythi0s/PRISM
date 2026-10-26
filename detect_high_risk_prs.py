import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import ast
import matplotlib.pyplot as plt

# Constants
CURRENT_DATE = datetime(2025, 5, 15)
CRITICAL_MODULES = ['auth', 'database', 'security']

AUTHORS = {
    'john.doe@example.com': 50,
    'jane.smith@example.com': 30,
    'mike.wilson@example.com': 20,
    'alex.brown@example.com': 15,
    'olivia.taylor@example.com': 10,
    'sarah.lee@example.com': 25,
    'liam.martin@example.com': 40,
    'james.moore@example.com': 35,
    'sophia.white@example.com': 18,
    'emma.jones@example.com': 22
}

pr_df = pd.read_csv("pull_requests.csv")
alignment_df = pd.read_csv("alignment_scores.csv")

# Helpers
def parse_files_changed(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []

def count_critical_files(files_changed):
    files = parse_files_changed(files_changed)
    return sum(1 for f in files if any(mod in f for mod in CRITICAL_MODULES))

merged_df = pr_df.merge(alignment_df[['pr_id', 'similarity_score']], on='pr_id', how='left')
merged_df['critical_file_count'] = merged_df['files_changed'].apply(count_critical_files)
merged_df['lines_modified'] = merged_df['lines_added'] + merged_df['lines_deleted']
merged_df['pr_age_days'] = (CURRENT_DATE - pd.to_datetime(merged_df['created_date'])).dt.days
merged_df['author_experience'] = merged_df['author'].map(AUTHORS).fillna(10)
merged_df['sbert_similarity_score'] = merged_df['similarity_score'].fillna(0.5)

# Regenerate is_high_risk labels using logic
merged_df['is_high_risk'] = merged_df.apply(lambda row: (
    (row['sbert_similarity_score'] < 0.7) +
    (row['lines_added'] > 120) +
    (row['lines_deleted'] > 80) +
    (row['critical_file_count'] > 1) +
    (row['comment_count'] == 0)
) >= 2, axis=1)

# Simplified feature set
features = [
    'critical_file_count', 'lines_added', 'lines_deleted', 'lines_modified',
    'comment_count', 'author_experience', 'sbert_similarity_score', 'avg_file_risk'
]

X = merged_df[features].fillna(0)
y = merged_df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train simple Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

importances = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nFeature Importance:")
print(importances.to_string(index=False))

# Save model
joblib.dump(rf_model, 'rf_high_risk_pr_model.pkl')

# Predict and classify all
merged_df['predicted_high_risk'] = rf_model.predict(X)
merged_df['high_risk_probability'] = rf_model.predict_proba(X)[:, 1]
merged_df['final_risk_label'] = merged_df['predicted_high_risk'].map({True: "High Risk", False: "Low Risk"})

# Save results
merged_df[['pr_id', 'final_risk_label', 'high_risk_probability'] + features].to_csv("pr_predictions.csv", index=False)
print("\n✅ Updated predictions saved to pr_predictions.csv")

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(importances['feature'], importances['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest Model')
plt.tight_layout()
plt.savefig("feature_importance_graph.png")
plt.show()
