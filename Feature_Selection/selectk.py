#Applying SelectK to the dataset lung_cancer_filter_mapped.csv for feature selection
#to get transformed dataset transformed_seelctk.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Load the dataset
df = pd.read_csv('lung_cancer_filter_mapped.csv')

# Separate features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Preprocessing pipeline for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply SelectKBest for feature selection, selecting around 30 features
selector = SelectKBest(score_func=f_classif, k=30)

# Create a pipeline that combines the preprocessor and the SelectKBest selector
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('selector', selector)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Get the feature names after preprocessing
preprocessed_feature_names = np.array(
    numeric_features.tolist() +
    list(pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))
)

# Get the selected features
selected_features = preprocessed_feature_names[selector.get_support()]

print("Selected features:", selected_features)

# Transform the entire dataset using the pipeline
X_transformed = pipeline.transform(X)

# Convert the transformed dataset to a DataFrame with the selected features
X_transformed_df = pd.DataFrame(X_transformed, columns=selected_features)

# Add the target variable to the transformed dataset
transformed_dataset = pd.concat([X_transformed_df, y.reset_index(drop=True)], axis=1)

# Save the transformed dataset to a CSV file
transformed_dataset.to_csv('transformed_selectk.csv', index=False)

print("Transformed dataset saved to 'transformed_selectk.csv'.")
