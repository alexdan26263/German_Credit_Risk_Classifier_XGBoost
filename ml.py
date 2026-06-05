import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid")

statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

y = y['class'].map({1: 0, 2: 1})

print("\nPreprocessing\n")

categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object', 'category', 'str']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Initializing scaler and encoder")
scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

print("Fitting and transforming train data")
X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_cols])

print("Transforming test data")
X_test_num_scaled = scaler.transform(X_test[numeric_cols])
X_test_cat_encoded = encoder.transform(X_test[categorical_cols])

X_train_processed = np.hstack((X_train_num_scaled, X_train_cat_encoded))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_encoded))

encoded_cat_names = encoder.get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(encoded_cat_names)


print("\nModel")

def evaluate_experiment(model, X_test_data, y_test_data, experiment_name):
    y_pred = model.predict(X_test_data)
    
    acc = accuracy_score(y_test_data, y_pred)
    prec = precision_score(y_test_data, y_pred)
    rec = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    cm = confusion_matrix(y_test_data, y_pred)
    
    print(f"\n{experiment_name}")
    print("-" * len(experiment_name))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Good (0)', 'Bad (1)'], 
                yticklabels=['Good (0)', 'Bad (1)'])
    plt.title(f'Confusion Matrix: {experiment_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    safe_name = experiment_name.replace(":", "").replace(" ", "_").replace("=", "_")
    plt.savefig(f"{safe_name}.png", bbox_inches='tight')
    plt.close()

def evaluate_with_threshold(model, X_test_data, y_test_data, experiment_name, threshold=0.5):
    probabilities = model.predict_proba(X_test_data)[:, 1] 

    y_pred_custom = (probabilities >= threshold).astype(int)
    
    acc = accuracy_score(y_test_data, y_pred_custom)
    prec = precision_score(y_test_data, y_pred_custom)
    rec = recall_score(y_test_data, y_pred_custom)
    f1 = f1_score(y_test_data, y_pred_custom)
    cm = confusion_matrix(y_test_data, y_pred_custom)
    
    print(f"\n{experiment_name} (Threshold: {threshold})")
    print("-" * len(experiment_name))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=['Good (0)', 'Bad (1)'], 
                yticklabels=['Good (0)', 'Bad (1)'])
    plt.title(f'Confusion Matrix: Threshold {threshold}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    safe_name = f"Experiment_5_Threshold_{str(threshold).replace('.', '_')}"
    plt.savefig(f"{safe_name}.png", bbox_inches='tight')
    plt.close()

print("\nExperiment 0: Baseline Model")
xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_base.fit(X_train_processed, y_train)
evaluate_experiment(xgb_base, X_test_processed, y_test, "Experiment 0: Default Parameters")

importances = xgb_base.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Top 10 featururi')
plt.savefig("Top_10_Feature.png", bbox_inches='tight')
plt.close()

print("\nExperiment 1: Max_depth Variance")
for depth in [3, 6, 10]:
    xgb_exp1 = XGBClassifier(max_depth=depth, random_state=42, eval_metric='logloss')
    xgb_exp1.fit(X_train_processed, y_train)
    evaluate_experiment(xgb_exp1, X_test_processed, y_test, f"Experiment 1: max_depth = {depth}")

print("\nExperiment 2: N_estimators Variance")
for n in [50, 100, 200]:
    xgb_exp2 = XGBClassifier(n_estimators=n, random_state=42, eval_metric='logloss')
    xgb_exp2.fit(X_train_processed, y_train)
    evaluate_experiment(xgb_exp2, X_test_processed, y_test, f"Experiment 2: n_estimators = {n}")

print("\nExperiment 3: Learning_rate Variance")
for lr in [0.01, 0.1, 0.3]:
    xgb_exp3 = XGBClassifier(learning_rate=lr, random_state=42, eval_metric='logloss')
    xgb_exp3.fit(X_train_processed, y_train)
    evaluate_experiment(xgb_exp3, X_test_processed, y_test, f"Experiment 3: learning_rate = {lr}")

print("\nExperiment 4: Optimized Model")


xgb_optimized = XGBClassifier(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.3,
    random_state=42, 
    eval_metric='logloss'
)

xgb_optimized.fit(X_train_processed, y_train)
evaluate_experiment(xgb_optimized, X_test_processed, y_test, "Experiment 4: Optimized Model")

print("\nExperiment 5: Advanced Imbalance Handling (SMOTE + Weights + Threshold)")

print("Applying SMOTE to training data")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"Original training target distribution:\n{y_train.value_counts()}")
print(f"SMOTE training target distribution:\n{y_train_smote.value_counts()}")

weight_ratio = (y_train_smote == 0).sum() / (y_train_smote == 1).sum() 

xgb_advanced = XGBClassifier(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.3,
    scale_pos_weight=weight_ratio,
    random_state=42, 
    eval_metric='logloss'
)

xgb_advanced.fit(X_train_smote, y_train_smote)
evaluate_with_threshold(xgb_advanced, X_test_processed, y_test, "Experiment 5: SMOTE + Default Threshold", threshold=0.5)
evaluate_with_threshold(xgb_advanced, X_test_processed, y_test, "Experiment 5: SMOTE + Lower Threshold", threshold=0.35)



print("\nGenerating ROC Curve")

y_prob = xgb_advanced.predict_proba(X_test_processed)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall / Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) - Experiment 5')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig("Experiment_5_ROC_Curve.png", bbox_inches='tight')
plt.show()