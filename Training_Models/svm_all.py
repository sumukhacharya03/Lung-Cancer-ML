#Applying SVM model to all the transformed datasets to get the accuracy and recall

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define a function to train and evaluate the SVM model
def evaluate_svm(dataset_path, dataset_name, target_column='label', kernel='linear'):
    print(f"Evaluating SVM on {dataset_name} dataset")
    print("=" * 50)
    
    # Load the transformed dataset
    data = pd.read_csv(dataset_path)
    
    # Separate features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Standardize the features and set up the SVM model in a pipeline
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, random_state=42))
    ])
    
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store the accuracy and recall for each fold
    accuracies = []
    recalls = []
    
    # Cross-validation loop
    for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model
        model_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model_pipeline.predict(X_test)
        
        # Calculate accuracy and recall
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Store results
        accuracies.append(accuracy)
        recalls.append(recall)
        
        # Print fold results
        print(f'Fold {fold}: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}')
    
    # Print average results
    print(f'\nAverage Accuracy: {sum(accuracies) / len(accuracies):.4f}')
    print(f'Average Recall: {sum(recalls) / len(recalls):.4f}')
    print("=" * 50)
    print("\n")

# Example usage with different datasets
evaluate_svm('transformed_pca.csv', 'PCA Transformed')
evaluate_svm('transformed_bso.csv', 'BSO Transformed')
evaluate_svm('transformed_rfe.csv', 'RFE Transformed')
evaluate_svm('transformed_selectk.csv', 'SelectK Transformed')
