# import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(df, target_col='class'):
    """Separate features and target, and perform train-test split."""
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest models."""
    # Initialize models
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    
    # Train models
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model


def evaluate_models(models, X_test, y_test):
    """Evaluate models using AUC-PR, F1-Score, and Confusion Matrix."""
    results = {}
    
    for name, model in models.items():
        # Predict probabilities for AUC-PR
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        
        # Predict labels for F1-Score and Confusion Matrix
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'AUC-PR': auc_pr,
            'F1-Score': f1,
            'Confusion Matrix': cm
        }
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.show()
    
    return results


def compare_models(results):
    """Compare models and determine the best based on evaluation metrics."""
    print("Model Evaluation Results:")
    best_model = None
    best_f1 = -1
    best_auc_pr = -1
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"AUC-PR: {metrics['AUC-PR']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")
        print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")
        
        # Determine best model based on F1-Score (primary) and AUC-PR (secondary)
        if metrics['F1-Score'] > best_f1 or (
            metrics['F1-Score'] == best_f1 and metrics['AUC-PR'] > best_auc_pr
        ):
            best_f1 = metrics['F1-Score']
            best_auc_pr = metrics['AUC-PR']
            best_model = name
    
    print(f"\nBest Model: {best_model}")
    print("Justification: The best model is chosen based on the highest F1-Score, "
          "as it balances precision and recall, which is critical for imbalanced data. "
          "AUC-PR is used as a secondary metric to evaluate performance on the minority class.")
    
    return best_model


def main(fraud_data):
    """Main function to prepare data, train, and evaluate models."""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(fraud_data, target_col='class')
    
    # Train models
    models = {
        'Logistic Regression': train_models(X_train, y_train)[0],
        'Random Forest': train_models(X_train, y_train)[1]
    }
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Compare models
    best_model = compare_models(results)
    
    return models, results, best_model


if __name__ == "__main__":
    # Example usage (assuming preprocessed data is available)
    # fraud_data = pd.read_csv('preprocessed_fraud_data.csv')
    # models, results, best_model = main(fraud_data)
    pass
