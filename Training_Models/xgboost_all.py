#Applying XG-Boost model to all the transformed datasets to get the accuracy and recall

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Load the dataset - assuming different filenames for different transformations
# Adjust filenames as per your actual dataset names
transformed_files = {
    'pca': 'transformed_pca.csv',
    'bso': 'transformed_bso.csv',
    'rfe': 'transformed_rfe.csv',
    'selectk': 'transformed_selectk.csv'
}

# Function to preprocess data based on transformation type
def preprocess_data(file, transformation_type):
    data = pd.read_csv(file)
    
    # Separate features and target
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Define preprocessing steps
    if transformation_type == 'pca' or transformation_type == 'bso':
        # Standardization for PCA and BSO
        numeric_features = X.columns
        numeric_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features)
            ])
    elif transformation_type == 'rfe':
        # Standardization for RFE
        numeric_features = X.columns
        numeric_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features)
            ])
    elif transformation_type == 'selectk':
        # Preprocessing for SelectK
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
    
    return X, y, preprocessor

# Define the XGBoost model
model = XGBClassifier(eval_metric='logloss')

# Define cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store results
results = {}

# Loop through each transformation type
for transformation_type, file in transformed_files.items():
    X, y, preprocessor = preprocess_data(file, transformation_type)
    
    # Create pipeline with preprocessing and XGBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Perform cross-validation
    accuracy_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    recall_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='recall')
    
    # Store results
    results[transformation_type] = {
        'accuracy': accuracy_scores,
        'recall': recall_scores,
        'mean_accuracy': accuracy_scores.mean(),
        'mean_recall': recall_scores.mean()
    }

# Print results for each transformation type
for transformation_type, result in results.items():
    print(f"Results for {transformation_type}:")
    print(f"Accuracy scores: {result['accuracy']}")
    print(f"Recall scores: {result['recall']}")
    print(f"Mean Accuracy: {result['mean_accuracy']}")
    print(f"Mean Recall: {result['mean_recall']}")
    print("---------------------------------------")
