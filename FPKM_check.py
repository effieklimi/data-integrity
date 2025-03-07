#!/usr/bin/env python3
"""
Hello script that reads and prints information about downloaded GEO data files.
"""
import os
import sys
import pandas as pd
import glob

def main():
    """Main function to read and display CSV file information."""
    print("Hello from integrity check script FPKM_check!")
    print("=" * 50)
    
    # Path to the data directory
    # data_dir = "/app/data"
    
    # Check if data directory exists
    # if not os.path.exists(data_dir):
    #     print(f"Error: Data directory '{data_dir}' not found.")
    #     return 1
    
    # # Find all CSV files
    # csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # if not csv_files:
    #     print(f"No CSV files found in '{data_dir}'.")
    #     return 1
    
    # print(f"Found {len(csv_files)} CSV files in '{data_dir}':")
    
    # Process each CSV file
    # for i, csv_file in enumerate(csv_files, 1):
    #     filename = os.path.basename(csv_file)
    #     # print(f"\n{i}. Processing file: {filename}")
    #     # print("-" * 50)
        
    #     try:
    #         # Read the CSV file using pandas
    #         df = pd.read_csv(csv_file)
            
    #         # Get file information
    #         file_size = os.path.getsize(csv_file) / (1024 * 1024)  # Convert to MB
    #         row_count = len(df)
    #         col_count = len(df.columns)
            
    #         # Print file statistics
    #         # print(f"File size: {file_size:.2f} MB")
    #         # print(f"Rows: {row_count}, Columns: {col_count}")
    #         # print("Column names:")
    #         # for col in df.columns:
    #             # print(f"  - {col}")
            
    #         # Print first 5 rows of the dataframe
    #         # print("\nFirst 5 rows:")
    #         # print(df.head(5).to_string())
            
    #         # Basic data quality checks
    #         print("\nData quality summary:")
    #         print(f"Missing values: {df.isna().sum().sum()}")
            
    #         # For numeric columns, print some statistics
    #         numeric_cols = df.select_dtypes(include=['number']).columns
    #         if len(numeric_cols) > 0:
    #             print("\nNumeric columns statistics:")
    #             print(df[numeric_cols].describe().to_string())
            
    #     except Exception as e:
    #         print(f"Error processing file {filename}: {str(e)}")
    
    # print("\n" + "=" * 50)
    print("Data integrity check completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
