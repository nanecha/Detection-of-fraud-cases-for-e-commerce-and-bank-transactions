# 📦 Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, confusion_matrix, precision_recall_curve,
    auc, classification_report
)
from sklearn.preprocessing import StandardScaler


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

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.', label=f'AUC={auc_pr:.2f}')
    plt.title(f'Precision-Recall Curve - {model_name} on {dataset_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"f1": f1, "auc_pr": auc_pr, "confusion_matrix": cm}

def split_data(creditcard, Fraud_Data):
    # Split CreditCard data
    X_cc = creditcard.drop(columns=['Class'])
    y_cc = creditcard['Class']
    X_cc_train, X_cc_test, y_cc_train, y_cc_test = train_test_split(
        X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=42)

    # Split Fraud_Data
    X_fd = Fraud_Data
    y_fd = pd.read_csv('F:/Detection-of-fraud-cases-for-e-commerce-and-bank-transactions/data/target_labels.csv')
    X_fd_train, X_fd_test, y_fd_train, y_fd_test = train_test_split(
        X_fd, y_fd, test_size=0.2, stratify=y_fd, random_state=42)
    #standardize the data
    scaler_cc = StandardScaler()
    X_cc_train_scaled = scaler_cc.fit_transform(X_cc_train)
    X_cc_test_scaled = scaler_cc.transform(X_cc_test)

    scaler_fd = StandardScaler()
    X_fd_train_scaled = scaler_fd.fit_transform(X_fd_train)
    X_fd_test_scaled = scaler_fd.transform(X_fd_test)
    return (X_cc_train_scaled, X_cc_test_scaled, y_cc_train, y_cc_test), \
           (X_fd_train_scaled, X_fd_test_scaled, y_fd_train, y_fd_test), \
           (X_cc_train, X_cc_test, y_cc_train, y_cc_test), \
              (X_fd_train, X_fd_test, y_fd_train, y_fd_test)            
              
              
              
def standardize_scaler(X_cc_train, X_cc_test, X_fd_train, X_fd_test):
    scaler_cc = StandardScaler()
    X_cc_train_scaled = scaler_cc.fit_transform(X_cc_train)
    X_cc_test_scaled = scaler_cc.transform(X_cc_test)

    scaler_fd = StandardScaler()
    X_fd_train_scaled = scaler_fd.fit_transform(X_fd_train)
    X_fd_test_scaled = scaler_fd.transform(X_fd_test)

    return (X_cc_train_scaled, X_cc_test_scaled), (X_fd_train_scaled, X_fd_test_scaled)


#  🚀 CREDITCARD DATASET
# logistic_regression_cc = LogisticRegression(max_iter=1000, random_state=42)


def train_model(X_cc_train_scaled, y_cc_train, X_cc_train, y_cc_test, 
                X_fd_train_scaled, y_fd_train, X_fd_train, y_fd_test):
    """
    Train Logistic Regression and Random Forest models on CreditCard and Fraud_Data datasets.
    """
    # Logistic Regression
    lr_cc = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_cc.fit(X_cc_train_scaled, y_cc_train)

    # Random Forest
    rf_cc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_cc.fit(X_cc_train, y_cc_train)
   # Logistic Regression
    lr_cc = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_cc.fit(X_cc_train_scaled, y_cc_train)
    #evaluate_model(lr_cc, X_cc_test_scaled, y_cc_test, 'CreditCard', 'Logistic Regression')

    # Random Forest
    rf_cc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_cc.fit(X_cc_train, y_cc_train)
    #evaluate_model(rf_cc, X_cc_test, y_cc_test, 'CreditCard', 'Random Forest')
    
    # Logistic Regression credit card
    lr_fd = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_fd.fit(X_fd_train_scaled, y_fd_train)
    #evaluate_model(lr_fd, X_fd_test_scaled, y_fd_test, 'Fraud_Data', 'Logistic Regression')

    # Random Forest fraud
    rf_fd = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_fd.fit(X_fd_train, y_fd_train)
    return lr_cc, rf_cc, lr_fd, rf_fd


