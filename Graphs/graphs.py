import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'Feature_Selection': ['RFE', 'RFE', 'RFE', 'RFE', 'SelectK', 'SelectK', 'SelectK', 'SelectK',
                          'PCA', 'PCA', 'PCA', 'PCA', 'BSO', 'BSO', 'BSO', 'BSO'],
    'Model': ['XGBoost', 'SVM', 'CatBoost', 'KNN', 'XGBoost', 'SVM', 'CatBoost', 'KNN',
              'XGBoost', 'SVM', 'CatBoost', 'KNN', 'XGBoost', 'SVM', 'CatBoost', 'KNN'],
    'Accuracy': [0.98776, 0.9846, 0.9878, 0.977, 0.98636, 0.9821, 0.9861, 0.9805, 
                 0.9779, 0.9776, 0.9799, 0.9457, 0.95962, 0.938, 0.962, 0.9182],
    'Recall': [0.96245, 0.9504, 0.9619, 0.977, 0.95975, 0.9422, 0.9583, 0.9321, 
               0.93334, 0.9391, 0.9407, 0.8153, 0.90334, 0.8459, 0.9104, 0.7896]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the desired order for the feature selection algorithms
feature_order = ['PCA', 'BSO', 'RFE', 'SelectK']

# Convert 'Feature_Selection' column to categorical with the specified order
df['Feature_Selection'] = pd.Categorical(df['Feature_Selection'], categories=feature_order, ordered=True)

# Plotting accuracy comparison using line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='Feature_Selection', y='Accuracy', hue='Model', data=df, marker='o')
plt.title('Accuracy Comparison Across Feature Selection Algorithms and Models')
plt.ylabel('Accuracy')
plt.xlabel('Feature Selection Algorithm')
plt.legend(title='Model')
plt.grid(True)
plt.ylim(0.9, 1)  # Set y-axis limits if needed
plt.show()

# Plotting recall comparison using line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='Feature_Selection', y='Recall', hue='Model', data=df, marker='o')
plt.title('Recall Comparison Across Feature Selection Algorithms and Models')
plt.ylabel('Recall')
plt.xlabel('Feature Selection Algorithm')
plt.legend(title='Model')
plt.grid(True)
plt.ylim(0.75, 1)  # Set y-axis limits if needed
plt.show()