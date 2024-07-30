#Applying PCA to the dataset lung_cancer_filter_mapped.csv for feature selection
#to get transformed dataset transformed_pca.csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load your dataset
df = pd.read_csv('lung_cancer_filter_mapped.csv')

# Identify the target column
target_column = 'label'

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Apply PCA
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95))
])

# Fit and transform the data
principal_components = pipeline.fit_transform(X)

# Create a new DataFrame with the principal components
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
df_pca = pd.DataFrame(principal_components, columns=pca_columns)

# Print the explained variance ratios
explained_variance_ratio = pipeline.named_steps['pca'].explained_variance_ratio_
print("Explained variance ratios for each principal component:")
print(explained_variance_ratio)

# Print the new dataset with principal components
print("\nNew dataset with principal components:")
print(df_pca)

# Save the new dataset
df_pca['label'] = y.values
df_pca.to_csv('transformed_pca.csv', index=False)
