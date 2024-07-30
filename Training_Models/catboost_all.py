#Applying CatBoost model to all the transformed datasets to get the accuracy and recall

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define a function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Split dataset into features and target
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipelines for numeric and categorical features
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    return X, y, preprocessor

# Function to train and evaluate the model
def evaluate_model(X, y, preprocessor):
    # Create a pipeline that includes preprocessing and CatBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(learning_rate=0.1, depth=6, iterations=500, verbose=0))
    ])
    
    # Define cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation
    accuracies = []
    recalls = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
    
    return accuracies, recalls

# File paths for the transformed datasets
file_paths = {
    'PCA': 'transformed_pca.csv',
    'BSO': 'transformed_bso.csv',
    'RFE': 'transformed_rfe.csv',
    'SelectK': 'transformed_selectk.csv'
}

# Evaluate models for each dataset
for name, file_path in file_paths.items():
    print(f'Evaluating for {name} transformed dataset:')
    X, y, preprocessor = load_and_preprocess_data(file_path)
    accuracies, recalls = evaluate_model(X, y, preprocessor)
    print(f'Accuracy: {sum(accuracies) / len(accuracies):.4f}')
    print(f'Recall: {sum(recalls) / len(recalls):.4f}\n')
