#Data Filtering: Dropping all columns that are empty, have same value,
#have less than 1000 boxes filled
#and named ptnum from lung_cancer.csv dataset
#and save it to a new dataset named lung_cancer_filter.csv

import pandas as pd

# Load the CSV file into a pandas DataFrame with low_memory=False
df = pd.read_csv('lung_cancer.csv', low_memory=False)

# Print total number of columns before any processing
initial_column_count = df.shape[1]
print(f"Total number of columns before processing: {initial_column_count}")

# Count and drop empty columns
empty_cols = df.columns[df.isnull().all()]
print(f"Number of empty columns: {len(empty_cols)}")
df.drop(empty_cols, axis=1, inplace=True)

# Count and drop columns with the same value in all rows
same_value_cols = df.columns[df.nunique() == 1]
print(f"Number of columns with the same value in all rows: {len(same_value_cols)}")
df.drop(same_value_cols, axis=1, inplace=True)

# Identify and drop columns with less than 1000 non-null values
cols_to_drop = df.columns[df.count() < 1000]
print(f"Number of columns with less than 1000 non-null values: {len(cols_to_drop)}")
df.drop(cols_to_drop, axis=1, inplace=True)

# Drop the column named 'ptnum' if it exists
if 'ptnum' in df.columns:
    df.drop('ptnum', axis=1, inplace=True)
    print("Column 'ptnum' has been removed.")
else:
    print("Column 'ptnum' does not exist in the DataFrame.")

# Print total number of columns after all filtering
final_column_count = df.shape[1]
print(f"Total number of columns after processing: {final_column_count}")

# Save the updated DataFrame to a new CSV file
df.to_csv('lung_cancer_filter.csv', index=False)