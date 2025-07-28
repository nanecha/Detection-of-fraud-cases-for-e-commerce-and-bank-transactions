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

## 6. Notes

* Be cautious of memory errors for large encoded feature matrices.
* Consider feature selection or dimensionality reduction for modeling.

---

## 7. Next Steps

* Model training and evaluation
* Real-time fraud detection implementation
* Model explainability (SHAP, LIME)

---

## Author

Nanecha Kebede

---

**📁 Path to run:** `src/data_preprocessing.py`

**📎 Output files:**

* `transformed_features.csv`
* `target_labels.csv`
* `eda_report.html`
