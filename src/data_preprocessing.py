import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
# from datetime import datetime


def handle_missing_values(df):
    """Handle missing values by imputing or dropping."""
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        # Impute numerical columns with median
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Impute categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def clean_data(df):
    """Remove duplicates and correct data types."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Ensure correct data types
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    if 'ip_address' in df.columns:
        df['ip_address'] = df['ip_address'].astype(float)
    return df


def perform_eda(df):
    """Perform univariate and bivariate analysis."""
    # Univariate analysis
    print("Univariate Analysis:")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            print(f"{col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, "
                  f"Min={df[col].min()}, Max={df[col].max()}")
        else:
            print(f"{col}: {df[col].value_counts().to_dict()}")
    
    # Bivariate analysis (correlation for numerical features)
    print("\nCorrelation Matrix:")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(df[numerical_cols].corr())
    
    return df


def merge_datasets(fraud_df, ip_df):
    """Merge fraud data with IP to country mapping."""
    # Ensure IP addresses are in float format
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(float)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float)
    
    # Function to find country for an IP address
    def find_country(ip, ip_df):
        for _, row in ip_df.iterrows():
            if row['lower_bound_ip_address'] <= ip <= row['upper_bound_ip_address']:
                return row['country']
        return 'Unknown'
    
    # Apply country mapping
    fraud_df['country'] = fraud_df['ip_address'].apply(lambda x: find_country(x, ip_df))
    return fraud_df


def engineer_features(df):
    """Create transaction frequency, velocity, and time-based features."""
    # Transaction frequency by user_id
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    
    # Transaction velocity (average time between transactions for each user)
    df['time_diff'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    df['avg_transaction_velocity'] = df.groupby('user_id')['time_diff'].transform('mean').fillna(0)
    
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0  # in hours
    
    return df


def transform_data(df, target_col='class', train=True):
    """Handle class imbalance, normalize/scale, 
    and encode categorical features."""
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col] if train else None
    
    # Define categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse=False), 
             categorical_cols)
        ])
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = list(numerical_cols) + list(cat_features)
        
    # Handle class imbalance (only for training data)
    if train and y is not None:
        smote = SMOTE(random_state=42)
        X_transformed, y = smote.fit_resample(X_transformed, y)
    
    # Convert back to DataFrame
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    
    if train:
        return X_transformed, y, preprocessor
    return X_transformed, preprocessor