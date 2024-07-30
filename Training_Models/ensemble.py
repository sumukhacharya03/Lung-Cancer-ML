import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Load your data
# Replace 'transformed_pca.csv' with the path to your dataset
data = pd.read_csv('transformed_selectk.csv')

# Prepare the dataset
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the base models
base_models = [
    ('xgb', XGBClassifier(eval_metric='logloss')),  # Removed use_label_encoder
    ('svm', SVC(probability=True)),
    ('catboost', CatBoostClassifier(verbose=0)),
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier())
]

# Create and evaluate each BaggingClassifier
results = {}
for name, model in base_models:
    # Create BaggingClassifier with the base model
    bagging_model = BaggingClassifier(estimator=model, n_estimators=10, random_state=42)
    
    # Fit the BaggingClassifier
    bagging_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = bagging_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store the results
    results[name] = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': cm
    }

# Print the results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
