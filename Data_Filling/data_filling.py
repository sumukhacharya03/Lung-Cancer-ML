#Data Filling: Filling up all the empty boxes using Random Forest for the lung_cancer_filter.csv dataset
#and save it to a new dataset named lung_cancer_filter_rf.csv

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('lung_cancer_filter.csv')

# Identify the target column
target_column = 'label'

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

def fill_missing_values_rf(df, col_name, numerical_cols, categorical_cols, model_type='regressor'):
    """
    Fill missing values in a specific column using RandomForest model.
    """
    df_notnull = df[df[col_name].notnull()]
    df_isnull = df[df[col_name].isnull()]

    if df_isnull.empty:
        print(f"No missing values found in column: {col_name}")
        return df

    print(f"Filling missing values for column: {col_name}")

    # Separate features and target for training the model
    X_train = df_notnull.drop(columns=[col_name])
    y_train = df_notnull[col_name]

    # Separate features for prediction
    X_test = df_isnull.drop(columns=[col_name])

    # Update numerical and categorical columns after dropping col_name
    numerical_cols = [col for col in numerical_cols if col != col_name]
    categorical_cols = [col for col in categorical_cols if col != col_name]

    # Preprocess categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
        ])

    if model_type == 'regressor':
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    else:
        # Encode target variable y if using classifier
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict the missing values
    if model_type == 'classifier':
        predictions = model.predict(X_test)
        predictions = label_encoder.inverse_transform(predictions)  # Inverse transform to original labels
    else:
        predictions = model.predict(X_test)

    df.loc[df[col_name].isnull(), col_name] = predictions

    return df

# Process numerical columns
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        print(f"Processing numerical column: {col}")
        X = fill_missing_values_rf(X, col, numerical_cols, categorical_cols, model_type='regressor')

# Process categorical columns
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        print(f"Processing categorical column: {col}")
        X = fill_missing_values_rf(X, col, numerical_cols, categorical_cols, model_type='classifier')

# Add the target column back to the DataFrame
X[target_column] = y

# Save the updated dataset to a new CSV file
X.to_csv('lung_cancer_filter_rf.csv', index=False)

# Display the first few rows of the updated DataFrame
print("Filled Data:")
print(X.head())

# Check if missing values have been filled
print("Number of missing values in each column after filling:")
print(X.isnull().sum())