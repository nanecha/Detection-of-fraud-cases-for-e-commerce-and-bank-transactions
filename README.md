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

## Author

Nanecha Kebede

---

**📁 Path to run:** `src/data_preprocessing.py`

**📎 Output files:**

* `transformed_features.csv`
* `target_labels.csv`
* `eda_report.html`

# 🧠 Task 2 – Model Building and Training

## 🔧 Objective

This task focuses on building and evaluating machine learning models to detect fraudulent e-commerce and bank transactions using preprocessed and engineered features obtained from Task 1.

---

## 📂 Step-by-Step Workflow

### 1. ✨ Data Loading and Preparation

* Load the previously processed data

### 2. ⚙️ Data Preprocessing

i Use Task 1 utilities to:

* Handle missing values
* Clean column formats and datatypes
* Merge with IP geolocation data
* Engineer time-based and frequency-based features

### 3. 📈 Feature Transformation

* Normalize/scale numerical features
* Encode categorical variables
* Address class imbalance using SMOTE or similar techniques

```python
X_transformed, y, preprocessor = transform_data(fraud_data, target_col='class', train=True)
```

Alternatively, use pre-saved files:

```python
processed_data = pd.read_csv('data/transformed_features.csv')
processed_data['class'] = pd.read_csv('data/target_labels.csv')
```

---

## 🤖 Model Training and Evaluation

### 4. 🔧 Model Selection

Train and evaluate multiple classifiers including:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost / LightGBM

The `main()` function in `src.model_training` handles model selection, training, and performance logging.

```python
models, results, best_model = main(processed_data)
```

### 5. ⚖️ Evaluation Metrics

Each model is assessed using:

* Accuracy
* Precision, Recall, F1 Score
* ROC-AUC
* Confusion Matrix

## 📊 Insights and Observations

* **Imbalance Issue**: Fraud class is heavily underrepresented; handled through oversampling.
* **Feature Importance**: Key contributors include:

  * `purchase_value`, `hour_of_day`, `time_diff`, `day_of_week`
  * Device/browser identifiers often correlated with fraud patterns.
* **Best Performing Models**: Tree-based ensemble models generally outperformed simple linear classifiers in terms of F1 and ROC-AUC.

---

## 📅 Outputs

* Trained models and performance results:

*This task builds on the cleaned and engineered dataset from Task 1. It forms the foundation for future deployment and model interpretability work.*
