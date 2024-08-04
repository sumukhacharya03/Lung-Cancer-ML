import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
import textwrap

# Load the dataset
file_path = 'lung_cancer_filter_mapped.csv'
dataset = pd.read_csv(file_path)

# Get the column names
columns = dataset.columns.tolist()

# Define a function to create and save a table figure
def create_table_image(columns, start_index, filename):
    fig, ax = plt.subplots(figsize=(30, 20))  # Larger figure size for better readability
    ax.axis('off')

    tbl = Table(ax, bbox=[0, 0, 1, 1])

    # Define width and height for cells
    width = 1.5
    height = 0.5  # Increased height for better readability

    # Wrap text for long column names
    max_col_width = 25  # Maximum width of column name before wrapping
    wrapped_columns = [textwrap.fill(col, max_col_width) for col in columns]

    # Add columns to the table
    num_cols = len(columns)
    num_rows = (num_cols + 1) // 2  # Calculate number of rows for each section

    for i in range(num_rows):
        col_num_left = i
        col_num_right = i + num_rows

        if col_num_left < num_cols:
            tbl.add_cell(i, 0, width=width, height=height, text=f"{start_index + col_num_left + 1}. {wrapped_columns[col_num_left]}", loc='left', 
                         facecolor='white', fontproperties={'weight': 'bold'})

        if col_num_right < num_cols:
            tbl.add_cell(i, 1, width=width, height=height, text=f"{start_index + col_num_right + 1}. {wrapped_columns[col_num_right]}", loc='left', 
                         facecolor='white', fontproperties={'weight': 'bold'})

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)  # Increased font size for better visibility
    tbl.scale(1.5, 1.5)   # Scale the table for better layout
    ax.add_table(tbl)

    # Save the figure
    plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0.1, facecolor='white')
    plt.close(fig)

# Number of columns per image
split_sizes = [23, 22, 22, 22]
start_idx = 0

# Create and save images
for i, size in enumerate(split_sizes):
    columns_subset = columns[start_idx:start_idx + size]
    create_table_image(columns_subset, start_idx, f'column_table_part_{i + 1}.png')
    start_idx += size
