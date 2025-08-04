# README.md

## Project: Detection of Fraud Cases for E-Commerce and Bank Transactions

This project involves a complete end-to-end data preprocessing and feature engineering pipeline for detecting fraud using multiple datasets. It includes data cleaning, exploratory data analysis (EDA), feature engineering, dataset merging, transformation, and saving processed data.

---

## 1. Setup Instructions

### Requirements

* Python 3.8+
* pandas, numpy
* scikit-learn
* seaborn, matplotlib
* ydata-profiling

### Project Structure

```
.
├──.github/
│   ├──workflows/ci.yml
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── transformed_features.csv
├──notesbook/   
│  └──EDA_and_Preprocessing.ipynb
│   ├── __init__.py
│   ├──model_building_training.ipynb
├── src/
│   └── data_preprocessing.py
├── test
├── Requirements.txt
├── .gitignore
├── README.md
```

---

## 2. Data Sources

* **Fraud\_Data.csv**: Main dataset with user and transaction information.
* **IpAddress\_to\_Country.csv**: IP geolocation mapping.
* **creditcard.csv**: Supplementary dataset for exploration.

---

## 3. Major Steps and Functions

### 1. Data Loading

```python
Fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
IP_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')
Creditcard_df = pd.read_csv('data/raw/creditcard.csv')
```

### 2. Handle Missing Values

```python
Fraud_df = handle_missing_values(Fraud_df)
IP_df = handle_missing_values(IP_df)
Creditcard_df = handle_missing_values(Creditcard_df)
```

### 3. Data Cleaning

* Remove duplicates
* Convert data types

```python
Fraud_df = clean_data(Fraud_df)
```

### 4. Exploratory Data Analysis

* Univariate & bivariate analysis
* Categorical vs Target analysis
* Correlation heatmap
* Automated profiling report using ydata-profiling

### 5. Merge Datasets for Geo-Analysis

```python
Fraud_df = merge_datasets_two(Fraud_df, IP_df)
```

### 6. Feature Engineering

```python
Fraud_df = engineer_features(Fraud_df)
```

* Time-based features: `hour_of_day`, `day_of_week`
* `time_since_signup`
* Transaction frequency & velocity

### 7. Data Transformation & Encoding

```python
X_transformed, y, preprocessor = transform_data_two(Fraud_df, target_col='class', train=True)
```

* Categorical encoding via OneHotEncoder
* Scaling numerical features
* Class rebalancing

---

## 4. Save Processed Data

```python
X_transformed_df.to_csv('data/transformed_features.csv', index=False)
pd.Series(y, name='class').to_csv('data/target_labels.csv', index=False)
```

---

## 5. EDA Profiling

```python
profile = ProfileReport(Fraud_df, title="Automated EDA Report")
profile.to_file("eda_report.html")
```



---

**📁 Path to run:** `src/data_preprocessing.py`

**📎 Output files:**

* `transformed_features.csv`
* `target_labels.csv`
* `eda_report.html`


# Task 2 - Model Selection and Training

This task focuses on selecting and training machine learning models to detect fraud using a previously preprocessed dataset. The objective is to evaluate different models and select the best-performing one based on evaluation metrics.

---

## 🔄 Workflow

1. **Data Loading and Preprocessing**
   - csv files  are read into pandas DataFrames.
   - Missing values are handled, and the data is cleaned.
   - IP-related features are merged with fraud transaction data.
   - Feature engineering is applied to generate meaningful features.
   - The data is transformed using encoding and scaling techniques to prepare for training.

2. **Dataset Preparation**
   - Transformed features and corresponding labels are loaded from CSV files.
   - The features and target are merged into a single dataset.

3. **Model Selection & Training**
   - The script runs multiple models and evaluates them using metrics such as F1-score, precision, recall, and AUC.
   - The best-performing model is selected based on these metrics.


## 🚀 Output

- Trained models and evaluation metrics.
- `best_model`: The highest-performing model ready for deployment or further interpretation (e.g., SHAP analysis).

---

## 💡 Next Steps

- Perform SHAP explainability analysis using the best model from this step.


# Task 3 - Model Explainability

This task focuses on explaining the predictions of the best-performing fraud detection model using SHAP (SHapley Additive exPlanations). The objective is to interpret what features most influence model decisions and generate both summary and individual explanation plots.

---

## 🔍 Objective

- Use SHAP to understand how features impact fraud prediction.
- Visualize feature importance using SHAP summary plots.
- Generate SHAP force plots to explain individual predictions.

---

## 📦 Imports and Modules

- `shap`: For SHAP value calculation and visualization.
- `matplotlib`, `seaborn`: For evaluation plots.
- `scikit-learn`: For training models and calculating metrics.
- `src.model_explain`: Contains reusable functions for preprocessing, training, evaluation, and SHAP visualization.

---

## 🧪 Process Overview

1. **Load Preprocessed Data**
   - Load `creditcard.csv`, `transformed_features.csv`, and `target_labels.csv`.

2. **Data Preprocessing**
   - Remove unnecessary columns like 'Time'.
   - Split into training and testing sets for both credit card and fraud datasets.
   - Scale features using `StandardScaler`.

3. **Model Training**
   - Train both Logistic Regression and Random Forest models.
   - Evaluate models using F1-score, AUC-PR, confusion matrix, and classification report.

4. **SHAP Analysis**
   - Use `TreeExplainer` on the best model (Random Forest).
   - Generate SHAP values for a sample of 100 rows.
   - Create:
     - **Summary Plot**: Displays global feature importance.
     - **Force Plot**: Visualizes SHAP values for an individual prediction.
   - Save the force plot as an HTML file (`force_plot.html`).

---

## 📁 Files Used

- `creditcard.csv`: Raw credit card transaction data.
- `transformed_features.csv`: Feature engineered fraud data.
- `target_labels.csv`: Target labels for fraud classification.

---

## 📊 Outputs

- Console output of evaluation metrics.
- Precision-Recall and Confusion Matrix plots.
- SHAP summary plot (bar plot of feature importance).
- `force_plot.html`: Interactive SHAP force plot saved for individual prediction analysis.

---

## 🧠 Insight

SHAP provides transparency into black-box models like Random Forest, which is crucial in financial fraud detection systems. Understanding which features drive predictions allows for better model validation and trust.

---

## ✅ Next Steps

- Use these visual insights to guide future feature engineering and model improvement.
- Optionally integrate SHAP explanations into a dashboard or reporting pipeline.



