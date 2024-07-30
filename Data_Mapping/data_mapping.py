#Data Mapping: Mapping codenames to the health parameters names from the lung_cancer_codes.csv file
#to the lung_cancer_filter_rf.csv dataset
#and save it to a new dataset named lung_cancer_filter_mapped.csv dataset

import pandas as pd

# Load the datasets from CSV files
df_codes = pd.read_csv('lung_cancer_filter_rf.csv')  # Replace with the path to your dataset with codes
df_mapping = pd.read_csv('lung_cancer_codes.csv')  # Replace with the path to your dataset with code-name mappings

# Assuming df_mapping has two columns: 'Code' and 'Name'
# Create a dictionary to map codes to names
code_to_name = dict(zip(df_mapping['code'], df_mapping['name']))

# Rename the columns in df_codes using the code_to_name dictionary
df_codes_renamed = df_codes.rename(columns=code_to_name)

# Save the new DataFrame to a new CSV file (optional)
df_codes_renamed.to_csv('lung_cancer_filter_mapped.csv', index=False)

# Display the renamed DataFrame (optional)
print(df_codes_renamed)