import pandas as pd
import numpy as np
import os

def normalize_first_column(input_filename, output_filename):
    """
    Reads a CSV file *with a header*, subtracts the second element (first data 
    row) of the first column from all subsequent elements in that column, 
    and saves the result to a new file.

    Args:
        input_filename (str): The path to the original CSV file.
        output_filename (str): The path where the updated CSV will be saved.
    """
    
    # 1. Check if the input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    # 2. Read the CSV file into a pandas DataFrame, treating the first row as the header
    try:
        # header=0 tells pandas that the first row is the header
        df = pd.read_csv(input_filename, header=0) 
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if the DataFrame is empty or only contains the header
    if df.empty:
        print("Error: The CSV contains only a header or is empty.")
        return

    # 3. Identify the first column using its name (pandas uses the value from the first row)
    # The name of the first column is df.columns[0]
    first_col_name = df.columns[0]
    
    # 4. Check if the column is numeric
    first_col_series = df[first_col_name]
    
    # Attempt to convert the column to a numeric type, forcing non-numeric (if any remain) to NaN
    numeric_col = pd.to_numeric(first_col_series, errors='coerce')
    
    # 5. Get the value of the **second element** (index 0 of the data rows)
    if numeric_col.size < 1:
        print("Error: The first column contains no data rows to process.")
        return

    # The first data element is at index 0 of the DataFrame data rows
    subtraction_value = numeric_col.iloc[0]
    
    if pd.isna(subtraction_value):
        print(f"Error: The first data element (second row, value='{first_col_series.iloc[0]}') is not a valid number and cannot be used for subtraction.")
        return

    print(f"Header: **{first_col_name}**")
    print(f"First data element (subtraction value): **{subtraction_value}**")
    
    # 6. Perform the subtraction operation on the numeric parts of the column
    # The operation is applied to the entire numeric column.
    df.loc[:, first_col_name] = numeric_col - subtraction_value

    print("Subtraction complete on the first column.")
    
    # 7. Write the updated DataFrame to a new CSV file
    # index=False prevents pandas from writing row numbers.
    # header=True (default) ensures the column names are preserved in the output.
    try:
        df.to_csv(output_filename, index=False)
        print(f"Successfully updated CSV saved to: **{output_filename}**")
    except Exception as e:
        print(f"Error writing to output file: {e}")


# --- Configuration ---
# IMPORTANT: Update these filenames before running
INPUT_FILE = 'GPS/heading.csv'
OUTPUT_FILE = 'GPS/normalized_heading.csv'
# ---------------------

if __name__ == "__main__":
    normalize_first_column(INPUT_FILE, OUTPUT_FILE)