#Applying KNN model to all the transformed datasets to get the accuracy and recall

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# File paths for each transformed dataset
datasets = {
    'PCA': 'transformed_pca.csv',
    'BSO': 'transformed_bso.csv',
    'RFE': 'transformed_rfe.csv',
    'SelectK': 'transformed_selectk.csv'
}

# Scorers
accuracy_scorer = make_scorer(accuracy_score)
recall_scorer = make_scorer(recall_score)

# KNN model definition
model = KNeighborsClassifier(n_neighbors=5)

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Preprocessing function
def preprocess_data(df):
    # Split into features and target
    X = df.drop(columns='label')
    y = df['label']
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessing pipelines
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
    
    return preprocessor, X, y

# Evaluate each dataset
for name, file_path in datasets.items():
    print(f"Evaluating {name} dataset")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Preprocess data
    preprocessor, X, y = preprocess_data(df)
    
    # Create pipeline with preprocessing and classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # SelectKBest only for 'SelectK' dataset
    if name == 'SelectK':
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
            ('classifier', model)
        ])
    
    # Perform cross-validation
    accuracy_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=accuracy_scorer)
    recall_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=recall_scorer)
    
    # Print results
    print(f"Accuracy scores: {accuracy_scores}")
    print(f"Recall scores: {recall_scores}")
    print(f"Mean Accuracy: {accuracy_scores.mean():.4f}")
    print(f"Mean Recall: {recall_scores.mean():.4f}")
    print("-" * 50)
