#!/usr/bin/env python3
"""
RNA-seq Quality Control Script - performs data integrity and quality control checks on RNA-seq count data.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.backends.backend_pdf import PdfPages
import time

def load_rnaseq_data(file_path):
    """Load RNA-seq count data from CSV file."""
    print(f"Loading data from: {file_path}")
    time.sleep(3) 
    try:
        # Assuming tab-separated file based on your example
        df = pd.read_csv(file_path, sep='\t')
        print(f"Successfully loaded data with {df.shape[0]} genes and {df.shape[1]} columns")
        time.sleep(3) 
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        time.sleep(3) 
        return None

def identify_sample_columns(df):
    """Identify sample columns (those containing count data)."""
    metadata_cols = ['ENSEMBL', 'name', 'type', 'chr', 'start', 'end', 'str']
    sample_cols = [col for col in df.columns if col not in metadata_cols]
    time.sleep(3) 
    print(f"Identified {len(sample_cols)} sample columns")
    time.sleep(3) 
    return metadata_cols, sample_cols

def check_library_size(df, sample_cols, output_dir):
    """Check total counts per sample to identify sequencing depth outliers."""
    time.sleep(3) 
    print("\n=== Checking Library Size / Sequencing Depth ===")
    time.sleep(3) 
    
    lib_sizes = df[sample_cols].sum()
    
    mean_size = lib_sizes.mean()
    std_size = lib_sizes.std()
    min_size = lib_sizes.min()
    max_size = lib_sizes.max()
    
    print(f"Mean library size: {mean_size:.2f}")
    time.sleep(3) 
    print(f"Library size std dev: {std_size:.2f}")
    time.sleep(3) 
    print(f"Minimum library size: {min_size:.2f} ({lib_sizes.idxmin()})")
    time.sleep(3) 
    print(f"Maximum library size: {max_size:.2f} ({lib_sizes.idxmax()})")
    time.sleep(3) 
    
    low_outliers = lib_sizes[lib_sizes < (mean_size - 2 * std_size)]
    high_outliers = lib_sizes[lib_sizes > (mean_size + 2 * std_size)]
    
    if len(low_outliers) > 0:
        print("\nPotential low-depth outliers:")
        time.sleep(3) 
        for sample, size in low_outliers.items():
            print(f"  {sample}: {size:.2f} counts")
            time.sleep(3) 
    
    if len(high_outliers) > 0:
        print("\nPotential high-depth outliers:")
        time.sleep(3) 
        for sample, size in high_outliers.items():
            print(f"  {sample}: {size:.2f} counts")
            time.sleep(3) 
    
    # Create a bar plot of library sizes
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=lib_sizes.index, y=lib_sizes.values)
    plt.title('Library Size per Sample')
    plt.ylabel('Total Counts')
    plt.xticks(rotation=90)
    
    # Add a horizontal line for the mean
    plt.axhline(y=mean_size, color='r', linestyle='--', label=f'Mean: {mean_size:.2f}')
    
    # Add lines for ±2 standard deviations
    plt.axhline(y=mean_size + 2*std_size, color='orange', linestyle=':', 
                label=f'Mean + 2SD: {mean_size + 2*std_size:.2f}')
    plt.axhline(y=max(0, mean_size - 2*std_size), color='orange', linestyle=':', 
                label=f'Mean - 2SD: {mean_size - 2*std_size:.2f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'library_sizes.png'))
    
    return lib_sizes

def check_count_distribution(df, sample_cols, output_dir):
    """Examine the distribution of counts within each sample."""
    time.sleep(3) 
    print("\n=== Checking Count Distribution ===")
    time.sleep(3) 
    
    # Create a PDF to store all plots
    pdf_path = os.path.join(output_dir, 'count_distributions.pdf')
    with PdfPages(pdf_path) as pdf:
        # Boxplot of log-transformed counts
        plt.figure(figsize=(14, 8))
        # Add small constant to avoid log(0)
        log_counts = np.log10(df[sample_cols] + 1)
        sns.boxplot(data=log_counts)
        plt.title('Distribution of log10(counts+1) per Sample')
        plt.ylabel('log10(counts+1)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Density plots (6 samples per figure)
        sample_groups = [sample_cols[i:i+6] for i in range(0, len(sample_cols), 6)]
        
        for i, group in enumerate(sample_groups):
            plt.figure(figsize=(10, 6))
            for sample in group:
                # Plot density of log counts (excluding zeros for clearer visualization)
                non_zero_counts = df[sample][df[sample] > 0]
                if len(non_zero_counts) > 0:  # Check if we have non-zero values
                    sns.kdeplot(np.log10(non_zero_counts), label=sample)
            
            plt.title(f'Density of log10(counts) (Non-zero values) - Group {i+1}')
            plt.xlabel('log10(counts)')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    print(f"Count distribution plots saved to {pdf_path}")
    time.sleep(3) 

def check_highly_expressed_genes(df, metadata_cols, sample_cols, output_dir):
    """Check if a small number of genes dominate the counts."""
    time.sleep(3) 
    print("\n=== Checking for Highly Expressed Genes ===")
    time.sleep(3) 
    
    # Calculate percentage of total counts per gene across all samples
    count_data = df[sample_cols]
    total_counts_per_sample = count_data.sum()
    gene_totals = count_data.sum(axis=1)
    total_counts = gene_totals.sum()
    
    gene_percentages = (gene_totals / total_counts) * 100
    
    # Combine with gene names for better reporting
    gene_counts_df = pd.DataFrame({
        'gene_id': df['ENSEMBL'],
        'gene_name': df['name'],
        'total_counts': gene_totals,
        'percentage': gene_percentages
    }).sort_values('percentage', ascending=False)
    
    # Report top 20 most expressed genes
    print("\nTop 20 most expressed genes:")
    time.sleep(3) 
    for i, (_, row) in enumerate(gene_counts_df.head(20).iterrows(), 1):
        print(f"{i}. {row['gene_name']} ({row['gene_id']}): {row['percentage']:.2f}% of total counts")
        time.sleep(3) 
    
    # Calculate cumulative percentage
    gene_counts_df['cumulative_percentage'] = gene_counts_df['percentage'].cumulative_sum()
    
    # Find how many genes account for 50% and 75% of counts
    genes_50pct = len(gene_counts_df[gene_counts_df['cumulative_percentage'] <= 50])
    genes_75pct = len(gene_counts_df[gene_counts_df['cumulative_percentage'] <= 75])
    
    total_genes = len(gene_counts_df)
    print(f"\n{genes_50pct} genes ({(genes_50pct/total_genes)*100:.2f}%) account for 50% of all counts")
    time.sleep(3) 
    print(f"{genes_75pct} genes ({(genes_75pct/total_genes)*100:.2f}%) account for 75% of all counts")
    time.sleep(3) 
    
    # Plot cumulative distribution of gene expression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(gene_counts_df) + 1), gene_counts_df['cumulative_percentage'])
    plt.title('Cumulative Distribution of Gene Expression')
    plt.xlabel('Number of Genes (ranked by expression)')
    plt.ylabel('Cumulative Percentage of Total Counts')
    plt.axhline(y=50, color='r', linestyle='--', label='50% of counts')
    plt.axhline(y=75, color='g', linestyle='--', label='75% of counts')
    plt.axvline(x=genes_50pct, color='r', linestyle=':')
    plt.axvline(x=genes_75pct, color='g', linestyle=':')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cumulative_expression.png'))
    
    # Create a pie chart of top 10 genes vs the rest
    top_genes = gene_counts_df.head(10)
    others_percentage = 100 - top_genes['percentage'].sum()
    
    plt.figure(figsize=(10, 8))
    labels = [f"{row['gene_name']} ({row['percentage']:.1f}%)" for _, row in top_genes.iterrows()]
    labels.append(f"All other genes ({others_percentage:.1f}%)")
    
    sizes = list(top_genes['percentage']) + [others_percentage]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Top 10 Most Expressed Genes vs All Others')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_genes_pie.png'))
    
    return gene_counts_df

def check_missing_values(df, metadata_cols, sample_cols):
    """Check for unexpected NAs or zeros in the data."""
    print("\n=== Checking for Missing Values ===")
    time.sleep(3) 
    
    # Check for NAs in the entire dataset
    na_counts = df.isna().sum()
    total_nas = na_counts.sum()
    
    if total_nas > 0:
        print(f"Found {total_nas} NA values in the dataset")
        time.sleep(3) 
        print("NA counts per column:")
        time.sleep(3) 
        for col, count in na_counts[na_counts > 0].items():
            print(f"  {col}: {count} NAs")
            time.sleep(3) 
    else:
        print("No NA values found in the dataset")
        time.sleep(3) 
    
    # Check for zeros in count data
    zeros_per_sample = (df[sample_cols] == 0).sum()
    total_genes = df.shape[0]
    
    print("\nZero counts per sample:")
    time.sleep(3) 
    for sample, zeros in zeros_per_sample.items():
        percentage = (zeros / total_genes) * 100
        print(f"  {sample}: {zeros} zeros ({percentage:.2f}% of genes)")
        time.sleep(3) 
    
    # Identify genes that are zero across all samples (might be problematic)
    all_zero_genes = df[(df[sample_cols] == 0).all(axis=1)]
    if len(all_zero_genes) > 0:
        print(f"\nFound {len(all_zero_genes)} genes with zero counts across all samples:")
        time.sleep(3) 
        for _, row in all_zero_genes.iterrows():
            print(f"  {row['name']} ({row['ENSEMBL']})")
            time.sleep(3) 
    else:
        print("\nNo genes with zero counts across all samples")
        time.sleep(3) 
    
    return zeros_per_sample, all_zero_genes

def main():
    """Main function to perform RNA-seq QC checks."""
    print("Starting RNA-seq Quality Control Script")
    time.sleep(3) 
    print("=" * 50)
    time.sleep(3) 
    
    # Path to the data directory
    data_dir = "/app/data"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        time.sleep(3) 
        return 1
    
    # Create output directory for plots and reports
    output_dir = os.path.join(data_dir, "qc_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        time.sleep(3) 
    
    # File path for the RNA-seq data
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{data_dir}'")
        return 1
    file_path = csv_files[0]  # Use the first CSV file found
        
    # Load data
    df = load_rnaseq_data(file_path)
    if df is None:
        return 1
    
    # Identify metadata and sample columns
    metadata_cols, sample_cols = identify_sample_columns(df)
    
    # Perform QC checks
    library_sizes = check_library_size(df, sample_cols, output_dir)
    check_count_distribution(df, sample_cols, output_dir)
    top_genes = check_highly_expressed_genes(df, metadata_cols, sample_cols, output_dir)
    zeros_info, all_zero_genes = check_missing_values(df, metadata_cols, sample_cols)
    
    # Generate a summary report
    report_path = os.path.join(output_dir, "qc_summary.txt")
    with open(report_path, 'w') as f:
        f.write("RNA-seq Quality Control Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset Information\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of genes: {df.shape[0]}\n")
        f.write(f"Number of samples: {len(sample_cols)}\n\n")
        
        f.write("Library Size Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean library size: {library_sizes.mean():.2f}\n")
        f.write(f"Library size std dev: {library_sizes.std():.2f}\n")
        f.write(f"Minimum library size: {library_sizes.min():.2f} ({library_sizes.idxmin()})\n")
        f.write(f"Maximum library size: {library_sizes.max():.2f} ({library_sizes.idxmax()})\n\n")
        
        f.write("Highly Expressed Genes\n")
        f.write("-" * 20 + "\n")
        f.write("Top 10 most expressed genes:\n")
        for i, (_, row) in enumerate(top_genes.head(10).iterrows(), 1):
            f.write(f"{i}. {row['gene_name']} ({row['gene_id']}): {row['percentage']:.2f}% of total counts\n")
        
        f.write("\nMissing Values Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total NA values: {df.isna().sum().sum()}\n")
        f.write(f"Genes with all zeros across samples: {len(all_zero_genes)}\n")
    
    print(f"\nQC summary report saved to {report_path}")
    time.sleep(3) 
    
    # Create JSON output with key QC metrics
    qc_data = {
        "dataset_info": {
            "num_genes": df.shape[0],
            "num_samples": len(sample_cols),
            "sample_names": sample_cols
        },
        "library_size": {
            "mean": float(library_sizes.mean()),
            "std_dev": float(library_sizes.std()),
            "min": float(library_sizes.min()),
            "min_sample": library_sizes.idxmin(),
            "max": float(library_sizes.max()),
            "max_sample": library_sizes.idxmax(),
            "per_sample": {sample: float(count) for sample, count in library_sizes.items()}
        },
        "highly_expressed_genes": {
            "top_10_genes": [
                {
                    "gene_id": row["gene_id"],
                    "gene_name": row["gene_name"],
                    "percentage": float(row["percentage"])
                }
                for _, row in top_genes.head(10).iterrows()
            ],
            "genes_for_50pct": int(len(top_genes[top_genes['cumulative_percentage'] <= 50])),
            "genes_for_75pct": int(len(top_genes[top_genes['cumulative_percentage'] <= 75]))
        },
        "missing_values": {
            "total_nas": int(df.isna().sum().sum()),
            "zero_counts": {
                "genes_with_all_zeros": int(len(all_zero_genes)),
                "zeros_per_sample": {sample: int(count) for sample, count in zeros_per_sample.items()},
                "zero_percentage_per_sample": {
                    sample: float((count / df.shape[0]) * 100) for sample, count in zeros_per_sample.items()
                }
            }
        }
    }
    
    # Write JSON to file
    json_path = os.path.join(output_dir, "qc_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(qc_data, f, indent=4)
    
    print(f"QC metrics JSON saved to {json_path}")
    time.sleep(3) 
    print("\nData quality control check completed successfully!")
    time.sleep(3) 
    return 0

if __name__ == "__main__":
    sys.exit(main())









# import os
# import sys
# import pandas as pd
# import glob

# def main():
#     """Main function to read and display CSV file information."""
#     print("Hello from integrity check script FPKM_check!")
#     print("=" * 50)
#     print("Data integrity check completed successfully!")
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())
