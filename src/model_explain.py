# 📁 src/model_explain.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, confusion_matrix, precision_recall_curve,
    auc, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import os

sns.set(style="whitegrid")


def preprocess_data(df, target_col, drop_cols=[]):
    df = df.drop(columns=drop_cols)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Split CreditCard
    X_cc = creditcard.drop(columns=['Class'])
    y_cc = creditcard['Class']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def standardize_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    else:
        raise ValueError("Invalid model_type. Use 'rf' or 'lr'.")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, dataset_name, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)

    print(f"\n📘 Evaluation Report for {model_name} on {dataset_name}")
    print("F1 Score:", round(f1, 4))
    print("AUC-PR:", round(auc_pr, 4))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.', label=f'AUC={auc_pr:.2f}')
    plt.title(f'Precision-Recall Curve - {model_name} on {dataset_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"f1": f1, "auc_pr": auc_pr, "confusion_matrix": cm}


def run_shap_analysis(model, X_test, X_train_columns, index=0, output_html="force_plot.html"):
    explainer = shap.TreeExplainer(model)

    X_sample = X_test[X_train_columns].sample(n=100, random_state=42)
    X_sample = X_sample.astype(X_test.dtypes.to_dict())

    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # Summary bar plot
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
    else:
        shap.summary_plot(shap_values, X_sample, plot_type="bar")

    # Force plot
    if isinstance(shap_values, list):
        row_shap = shap_values[1][index]
        base_value = explainer.expected_value[1]
    else:
        row_shap = shap_values[index]
        base_value = explainer.expected_value

    shap_html = shap.force_plot(base_value, row_shap, X_sample.iloc[index], matplotlib=False)
    shap.save_html(output_html, shap_html)
    print(f"✅ SHAP force plot saved to {output_html}")
    return shap_html

# Example usage

if __name__ == "__main__":
    pass
