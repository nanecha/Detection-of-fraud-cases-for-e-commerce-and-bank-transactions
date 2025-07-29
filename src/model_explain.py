import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.model_training import prepare_data, train_models


def generate_shap_plots(X_train, X_test, model, model_name, output_dir='shap_plots'):
    """Generate SHAP Summary and Force plots for model explainability."""
    # Initialize SHAP explainer for tree-based model
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification (use class 1 SHAP values for fraud)
    if len(shap_values) == 2:
        shap_values = shap_values[1]
    
    # Summary Plot (Global feature importance)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    
    # Force Plot for first test instance (Local feature importance)
    shap.initjs()  # Initialize JavaScript for force plot
    shap.force_plot(
        explainer.expected_value[1] if len(explainer.expected_value) == 2 else explainer.expected_value,
        shap_values[0],
        X_test.iloc[0],
        show=False,
        matplotlib=True
    )
    plt.title(f"SHAP Force Plot - First Test Instance - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_force_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    
    return shap_values


def interpret_shap_results(shap_values, X_test):
    """Generate text interpretation of SHAP results for the report."""
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap_value': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='mean_shap_value', ascending=False)
    
    interpretation = [
        "\\subsection{SHAP Analysis Results}",
        "\\textbf{Global Feature Importance (Summary Plot):}",
        "The SHAP Summary Plot illustrates the impact of each feature on the model's predictions for fraudulent transactions. Features are ranked by their mean absolute SHAP values, indicating their overall contribution to predicting fraud. The top features driving fraud predictions are:"
    ]
    
    # Add top 5 features
    for i, row in feature_importance.head(5).iterrows():
        interpretation.append(f"\\item \\texttt{{{row['feature']}}}: Mean SHAP value = {row['mean_shap_value']:.4f}. "
                            f"This feature significantly influences fraud predictions, with higher/lower values "
                            f"associated with {'increased' if 'time_since_signup' in row['feature'] or 'transaction_count' in row['feature'] else 'varying'} fraud likelihood.")
    
    interpretation.extend([
        "",
        "\\textbf{Local Feature Importance (Force Plot):}",
        "The SHAP Force Plot for the first test instance shows how individual feature values contribute to the model's prediction for that specific transaction. Features pushing the prediction toward fraud (positive SHAP values) or away from fraud (negative SHAP values) are highlighted. For example, a low \\texttt{time_since_signup} may strongly increase the fraud probability, indicating rapid transactions post-signup are suspicious.",
        "",
        "\\textbf{Key Drivers of Fraud:}",
        "The SHAP analysis reveals that features like \\texttt{time_since_signup}, \\texttt{transaction_count}, and \\texttt{country} are critical drivers of fraud. A short time between signup and purchase often indicates fraudulent behavior, as fraudsters may create accounts solely for quick transactions. High transaction counts suggest repeated activity, potentially automated. Certain countries may have higher fraud prevalence, possibly due to regional patterns or data biases."
    ])
    
    return "\n".join(interpretation)


def main(fraud_data, model_name='Random Forest'):
    """Main function to generate SHAP plots and interpretations."""
    # Ensure output directory exists
    import os
    os.makedirs('shap_plots', exist_ok=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(fraud_data, target_col='class')
    
    # Train models and select the specified model
    models = {
        'Logistic Regression': train_models(X_train, y_train)[0],
        'Random Forest': train_models(X_train, y_train)[1]
    }
    model = models[model_name]
    
    # Generate SHAP plots
    shap_values = generate_shap_plots(X_train, X_test, model, model_name)
    
    # Generate interpretation for report
    interpretation = interpret_shap_results(shap_values, X_test)
    
    return shap_values, interpretation


if __name__ == "__main__":
    # Example usage (assuming preprocessed data is available)
    # fraud_data = pd.read_csv('preprocessed_fraud_data.csv')
    # shap_values, interpretation = main(fraud_data)
    
    pass