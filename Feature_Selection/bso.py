#Applying BSO to the dataset lung_cancer_filter_mapped.csv for feature selection
#to get transformed dataset transformed_bso.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import random

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
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Define fitness function
def fitness(solution, X, y):
    selected_features = np.where(solution == 1)[0]
    if len(selected_features) == 0:
        return 0  # Avoid empty feature selection
    X_selected = X[:, selected_features]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

# Initialize population
def initialize_population(pop_size, num_features):
    return [np.random.randint(2, size=num_features) for _ in range(pop_size)]

# Brain Storm Optimization algorithm
def BSO(X, y, pop_size=10, num_iterations=50):
    num_features = X.shape[1]
    population = initialize_population(pop_size, num_features)
    best_solution = None
    best_fitness = -1

    for iteration in range(num_iterations):
        new_population = []
        for solution in population:
            new_solution = solution.copy()
            if random.random() < 0.5:
                # Mutation
                idx = random.randint(0, num_features - 1)
                new_solution[idx] = 1 - new_solution[idx]
            else:
                # Crossover
                partner = random.choice(population)
                crossover_point = random.randint(0, num_features - 1)
                new_solution[:crossover_point] = partner[:crossover_point]

            new_fitness = fitness(new_solution, X, y)
            if new_fitness > best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness

            new_population.append(new_solution)
        
        population = new_population
        print(f"Iteration {iteration+1}/{num_iterations}, Best Fitness: {best_fitness}")

    return best_solution, best_fitness

# Run BSO
best_solution, best_fitness = BSO(X_preprocessed, y)

# Get selected features
selected_features = np.where(best_solution == 1)[0]
selected_feature_names = np.array(preprocessor.get_feature_names_out())[selected_features]

# Print selected features
print("Selected features:")
print(selected_feature_names)

# Create the final dataset with selected features
X_final = X_preprocessed[:, selected_features]

# Convert final dataset to DataFrame for better readability
selected_feature_names_df = np.array(preprocessor.get_feature_names_out())[selected_features]
df_final = pd.DataFrame(X_final, columns=selected_feature_names_df)

# Optionally, add the target column back to the final dataset
df_final[target_column] = y.values

# Print or save the final dataset
print("\nFinal dataset with selected features:")
print(df_final.head())

# Optionally, save the final dataset to a CSV file
df_final.to_csv('transformed_bso.csv', index=False)
