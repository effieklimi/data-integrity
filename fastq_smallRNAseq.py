#!/usr/bin/env python3
"""
small_rna_seq_qc.py - Quality Control for small RNA-Seq FASTQ files

This script performs robust data integrity and quality control checks on FASTQ files
from small RNA sequencing experiments, including:
- File format validation
- Read quality metrics (Phred scores)
- Adapter contamination detection
- Length distribution analysis (critical for small RNA studies)
- Sequence complexity assessment
- Potential contamination checks
- Basic summary statistics

Usage:
    python small_rna_seq_qc.py -i input.fastq [-o output_directory] [options]

Requirements:
    - Python 3.6+
    - Biopython
    - Matplotlib
    - Pandas
    - NumPy
"""

import os
import sys
import gzip
import argparse
import logging
from collections import defaultdict, Counter
import re
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqUtils import GC

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common adapter sequences for small RNA-seq
ADAPTERS = {
    "Illumina_TruSeq": "TGGAATTCTCGGGTGCCAAGG",
    "NEBNext_small_RNA": "AGATCGGAAGAGCACACGTCT",
    "NEXTflex_small_RNA": "TGGAATTCTCGGGTGCCAAGGC",
}

# Common small RNA sizes (in nucleotides)
SMALL_RNA_SIZES = {
    "miRNA": (19, 25),
    "piRNA": (24, 32),
    "siRNA": (20, 24),
    "snoRNA": (60, 300),
    "tRNA_fragment": (15, 40),
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quality control for small RNA-seq FASTQ files")
    
    parser.add_argument("-i", "--input", required=True, help="Input FASTQ file(s)", nargs='+')
    parser.add_argument("-o", "--output", default="./qc_output", help="Output directory")
    parser.add_argument("-t", "--threads", type=int, default=mp.cpu_count(), 
                        help="Number of threads to use")
    parser.add_argument("--min-length", type=int, default=15, 
                        help="Minimum read length to consider")
    parser.add_argument("--max-length", type=int, default=50, 
                        help="Maximum read length to consider")
    parser.add_argument("--sample-size", type=int, default=1000000, 
                        help="Number of reads to sample (0 for all)")
    parser.add_argument("--phred-encoding", choices=["phred33", "phred64"], default="phred33", 
                        help="FASTQ quality score encoding")
    parser.add_argument("--adapter-check", action="store_true", 
                        help="Check for adapter contamination")
    parser.add_argument("--custom-adapter", help="Custom adapter sequence to check")
    
    return parser.parse_args()

def is_gzipped(filename):
    """Check if a file is gzipped based on its magic number."""
    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def open_fastq(filename):
    """Open a FASTQ file, gzipped or not."""
    if is_gzipped(filename):
        return gzip.open(filename, 'rt')
    else:
        return open(filename, 'r')

def calculate_phred_scores(record, phred_encoding):
    """Calculate quality scores from FASTQ quality string."""
    if phred_encoding == "phred33":
        offset = 33
    else:  # phred64
        offset = 64
    
    quality_scores = [ord(q) - offset for q in record.letter_annotations["phred_quality"]]
    return quality_scores

def check_file_integrity(fastq_file):
    """Verify FASTQ file integrity by checking that it's a valid FASTQ file."""
    try:
        with open_fastq(fastq_file) as f:
            # Check first 10 records for proper FASTQ format
            for i, record in enumerate(SeqIO.parse(f, "fastq")):
                if i >= 10:
                    break
                # Check for sequence and quality score length match
                if len(record.seq) != len(record.letter_annotations["phred_quality"]):
                    return False, "Sequence length and quality score length mismatch"
                # Check for valid nucleotides
                if not all(n in 'ACGTN' for n in str(record.seq).upper()):
                    return False, "Invalid nucleotides found"
        return True, "File appears to be valid FASTQ"
    except Exception as e:
        return False, f"Error checking file integrity: {str(e)}"

def count_reads(fastq_file):
    """Count the total number of reads in a FASTQ file."""
    try:
        count = 0
        with open_fastq(fastq_file) as f:
            for _ in SeqIO.parse(f, "fastq"):
                count += 1
        return count
    except Exception as e:
        logger.error(f"Error counting reads: {str(e)}")
        return 0

def check_adapter_contamination(records, adapters, contamination_threshold=0.9):
    """Check for adapter contamination in reads."""
    adapter_counts = defaultdict(int)
    total_reads = len(records)
    
    if total_reads == 0:
        return {}
    
    for adapter_name, adapter_seq in adapters.items():
        for record in records:
            seq = str(record.seq)
            if adapter_seq in seq:
                adapter_counts[adapter_name] += 1
    
    # Calculate contamination percentages
    results = {}
    for adapter_name, count in adapter_counts.items():
        percentage = (count / total_reads) * 100
        results[adapter_name] = {
            "count": count,
            "percentage": percentage,
            "is_contaminated": percentage > contamination_threshold
        }
    
    return results

def calculate_sequence_complexity(seq):
    """Calculate sequence complexity using Shannon entropy."""
    if not seq:
        return 0
    
    base_counts = Counter(seq)
    length = len(seq)
    entropy = 0
    
    for count in base_counts.values():
        frequency = count / length
        entropy -= frequency * np.log2(frequency)
    
    # Normalize by maximum possible entropy (2 for DNA/RNA)
    return entropy / 2

def detect_potential_contamination(records):
    """
    Detect potential contaminations by checking for:
    - Overrepresented sequences
    - Unusual GC content
    - Repetitive sequences
    """
    sequence_counts = Counter()
    gc_contents = []
    complexity_scores = []
    
    for record in records:
        sequence = str(record.seq)
        sequence_counts[sequence] += 1
        gc_contents.append(GC(sequence))
        complexity_scores.append(calculate_sequence_complexity(sequence))
    
    # Find overrepresented sequences (appearing in more than 0.1% of reads)
    total_reads = len(records)
    threshold = total_reads * 0.001
    overrepresented = {seq: count for seq, count in sequence_counts.most_common(10) 
                       if count > threshold}
    
    # Calculate GC content stats
    gc_mean = np.mean(gc_contents)
    gc_std = np.std(gc_contents)
    
    # Calculate complexity stats
    complexity_mean = np.mean(complexity_scores)
    complexity_std = np.std(complexity_scores)
    
    return {
        "overrepresented_sequences": overrepresented,
        "gc_content": {
            "mean": gc_mean,
            "std": gc_std,
            "unusual": gc_mean < 30 or gc_mean > 70  # Unusual GC content thresholds
        },
        "sequence_complexity": {
            "mean": complexity_mean,
            "std": complexity_std,
            "low_complexity": complexity_mean < 0.6  # Low complexity threshold
        }
    }

def analyze_length_distribution(records, small_rna_sizes):
    """Analyze read length distribution focusing on small RNA ranges."""
    lengths = [len(record.seq) for record in records]
    
    if not lengths:
        return {}
    
    length_counts = Counter(lengths)
    
    # Calculate basic statistics
    stats = {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "std": np.std(lengths),
        "counts": dict(length_counts)
    }
    
    # Calculate counts within each small RNA size range
    for rna_type, (min_len, max_len) in small_rna_sizes.items():
        in_range = sum(1 for length in lengths if min_len <= length <= max_len)
        stats[f"{rna_type}_count"] = in_range
        stats[f"{rna_type}_percentage"] = (in_range / len(lengths)) * 100
    
    return stats

def analyze_quality_scores(records, phred_encoding):
    """Analyze quality scores across positions."""
    if not records:
        return {}
    
    # Get the length of the longest read
    max_length = max(len(record.seq) for record in records)
    
    # Initialize arrays to store quality scores and counts
    total_scores = np.zeros(max_length)
    counts = np.zeros(max_length, dtype=int)
    
    # Sum quality scores
    for record in records:
        quality_scores = calculate_phred_scores(record, phred_encoding)
        length = len(quality_scores)
        total_scores[:length] += quality_scores
        counts[:length] += 1
    
    # Calculate mean quality per position
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_quality = np.divide(total_scores, counts)
        mean_quality = np.nan_to_num(mean_quality)
    
    # Calculate quality score distribution
    all_scores = []
    for record in records:
        all_scores.extend(calculate_phred_scores(record, phred_encoding))
    
    quality_distribution = Counter(all_scores)
    
    return {
        "per_position_quality": {i: score for i, score in enumerate(mean_quality)},
        "quality_distribution": dict(quality_distribution),
        "mean_quality": np.mean(all_scores),
        "min_quality": np.min(all_scores) if all_scores else 0
    }

def plot_length_distribution(length_stats, output_file):
    """Plot read length distribution."""
    plt.figure(figsize=(10, 6))
    
    counts = length_stats["counts"]
    lengths = sorted(counts.keys())
    values = [counts[length] for length in lengths]
    
    plt.bar(lengths, values, color='steelblue')
    
    # Add markers for common small RNA types
    for rna_type, (min_len, max_len) in SMALL_RNA_SIZES.items():
        mid_point = (min_len + max_len) / 2
        max_count = max(values) * 0.9
        plt.annotate(rna_type, xy=(mid_point, max_count * (0.5 + list(SMALL_RNA_SIZES.keys()).index(rna_type) * 0.1)), 
                     ha='center', va='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
        
        # Draw range
        plt.axvspan(min_len, max_len, alpha=0.1, color='green')
    
    plt.title('Read Length Distribution')
    plt.xlabel('Read Length (nt)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_quality_heatmap(quality_stats, output_file):
    """Plot quality scores as a heatmap."""
    per_position_quality = quality_stats["per_position_quality"]
    positions = sorted(per_position_quality.keys())
    qualities = [per_position_quality[pos] for pos in positions]
    
    plt.figure(figsize=(12, 6))
    
    # Create a heatmap-like display
    plt.bar(positions, qualities, color=[quality_to_color(q) for q in qualities], width=1.0, edgecolor='none')
    
    # Color guidelines
    plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7)
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    
    plt.title('Quality Scores Across All Bases')
    plt.xlabel('Position in Read (bp)')
    plt.ylabel('Mean Quality Score')
    plt.ylim(0, max(40, max(qualities) + 5))
    
    # Add color legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color='orange', alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7)
    ]
    labels = ['Poor quality (<20)', 'Reasonable quality (20-30)', 'Good quality (>30)']
    plt.legend(handles, labels, loc='upper right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def quality_to_color(quality):
    """Convert quality score to color for visualization."""
    if quality < 20:
        return 'red'
    elif quality < 30:
        return 'orange'
    else:
        return 'green'

def analyze_fastq(args):
    """Main function to analyze FASTQ files."""
    fastq_file, options = args
    results = {"filename": os.path.basename(fastq_file)}

    try:
        # Check file integrity
        is_valid, message = check_file_integrity(fastq_file)
        results["is_valid"] = is_valid
        results["validation_message"] = message
        
        if not is_valid:
            logger.error(f"File {fastq_file} failed integrity check: {message}")
            return results
        
        # Count total reads
        total_reads = count_reads(fastq_file)
        results["total_reads"] = total_reads
        
        if total_reads == 0:
            logger.error(f"No reads found in {fastq_file}")
            return results
        
        # Sample reads for analysis
        sample_size = options.sample_size if options.sample_size > 0 else total_reads
        sample_size = min(sample_size, total_reads)
        
        # Load sampled records
        sampled_records = []
        with open_fastq(fastq_file) as f:
            all_records = list(SeqIO.parse(f, "fastq"))
            
            if sample_size < total_reads:
                # Randomly sample records
                indices = np.random.choice(total_reads, sample_size, replace=False)
                sampled_records = [all_records[i] for i in indices]
            else:
                sampled_records = all_records
        
        logger.info(f"Analyzing {len(sampled_records)} reads from {fastq_file}")
        
        # Filter reads by length
        filtered_records = [r for r in sampled_records 
                           if options.min_length <= len(r.seq) <= options.max_length]
        results["records_in_size_range"] = len(filtered_records)
        results["percentage_in_size_range"] = (len(filtered_records) / len(sampled_records)) * 100
        
        # Analyze read lengths
        length_stats = analyze_length_distribution(filtered_records, SMALL_RNA_SIZES)
        results["length_stats"] = length_stats
        
        # Analyze quality scores
        quality_stats = analyze_quality_scores(filtered_records, options.phred_encoding)
        results["quality_stats"] = quality_stats
        
        # Check for potential contamination
        contamination = detect_potential_contamination(filtered_records)
        results["contamination"] = contamination
        
        # Check for adapter contamination if requested
        if options.adapter_check:
            adapters_to_check = ADAPTERS.copy()
            if options.custom_adapter:
                adapters_to_check["custom"] = options.custom_adapter
                
            adapter_results = check_adapter_contamination(filtered_records, adapters_to_check)
            results["adapter_contamination"] = adapter_results
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing {fastq_file}: {str(e)}")
        results["error"] = str(e)
        return results

def generate_report(results, output_dir):
    """Generate an HTML report with the QC results."""
    report_file = os.path.join(output_dir, "qc_report.html")
    
    with open(report_file, 'w') as f:
        # Start HTML report
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Small RNA-Seq QC Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
                .warning { color: #e74c3c; }
                .pass { color: #2ecc71; }
                .file { margin-bottom: 40px; border-bottom: 1px solid #ddd; padding-bottom: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Small RNA-Seq Quality Control Report</h1>
            <div class="summary">
                <p>Generated on: %s</p>
                <p>Total files analyzed: %d</p>
            </div>
        """ % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(results)))
        
        # For each file
        for result in results:
            filename = result["filename"]
            f.write(f"<div class='file'><h2>File: {filename}</h2>")
            
            # Validation status
            status_class = "pass" if result.get("is_valid", False) else "warning"
            status_msg = result.get("validation_message", "No validation performed")
            f.write(f"<p>Validation: <span class='{status_class}'>{status_msg}</span></p>")
            
            # Basic stats
            f.write("<h3>Basic Statistics</h3>")
            f.write("<table>")
            f.write("<tr><th>Metric</th><th>Value</th></tr>")
            f.write(f"<tr><td>Total reads</td><td>{result.get('total_reads', 'N/A')}</td></tr>")
            f.write(f"<tr><td>Reads in size range</td><td>{result.get('records_in_size_range', 'N/A')}</td></tr>")
            f.write(f"<tr><td>Percentage in size range</td><td>{result.get('percentage_in_size_range', 'N/A'):.2f}%</td></tr>")
            
            # Length stats
            length_stats = result.get("length_stats", {})
            if length_stats:
                f.write(f"<tr><td>Mean read length</td><td>{length_stats.get('mean', 'N/A'):.2f}</td></tr>")
                f.write(f"<tr><td>Median read length</td><td>{length_stats.get('median', 'N/A')}</td></tr>")
                f.write(f"<tr><td>Min read length</td><td>{length_stats.get('min', 'N/A')}</td></tr>")
                f.write(f"<tr><td>Max read length</td><td>{length_stats.get('max', 'N/A')}</td></tr>")
                
                # Small RNA content
                for rna_type in SMALL_RNA_SIZES.keys():
                    count_key = f"{rna_type}_count"
                    pct_key = f"{rna_type}_percentage"
                    if count_key in length_stats:
                        f.write(f"<tr><td>{rna_type} reads</td><td>{length_stats[count_key]} ({length_stats[pct_key]:.2f}%)</td></tr>")
            
            # Quality stats
            quality_stats = result.get("quality_stats", {})
            if quality_stats:
                mean_quality = quality_stats.get("mean_quality", 0)
                quality_class = "warning" if mean_quality < 30 else "pass"
                f.write(f"<tr><td>Mean quality score</td><td class='{quality_class}'>{mean_quality:.2f}</td></tr>")
                f.write(f"<tr><td>Min quality score</td><td>{quality_stats.get('min_quality', 'N/A')}</td></tr>")
            
            f.write("</table>")
            
            # Contamination
            contamination = result.get("contamination", {})
            if contamination:
                f.write("<h3>Potential Contamination</h3>")
                
                # GC content
                gc_info = contamination.get("gc_content", {})
                gc_class = "warning" if gc_info.get("unusual", False) else "pass"
                f.write(f"<p>GC Content: <span class='{gc_class}'>{gc_info.get('mean', 0):.2f}%</span></p>")
                
                # Sequence complexity
                complexity = contamination.get("sequence_complexity", {})
                complexity_class = "warning" if complexity.get("low_complexity", False) else "pass"
                f.write(f"<p>Sequence Complexity: <span class='{complexity_class}'>{complexity.get('mean', 0):.2f}</span></p>")
                
                # Overrepresented sequences
                overrep = contamination.get("overrepresented_sequences", {})
                if overrep:
                    f.write("<h4>Overrepresented Sequences</h4>")
                    f.write("<table><tr><th>Sequence</th><th>Count</th></tr>")
                    for seq, count in overrep.items():
                        truncated_seq = seq[:20] + "..." if len(seq) > 20 else seq
                        f.write(f"<tr><td>{truncated_seq}</td><td>{count}</td></tr>")
                    f.write("</table>")
            
            # Adapter contamination
            adapter_results = result.get("adapter_contamination", {})
            if adapter_results:
                f.write("<h3>Adapter Contamination</h3>")
                f.write("<table><tr><th>Adapter</th><th>Percentage</th><th>Status</th></tr>")
                for adapter, info in adapter_results.items():
                    status_class = "warning" if info.get("is_contaminated", False) else "pass"
                    status_text = "Contaminated" if info.get("is_contaminated", False) else "OK"
                    f.write(f"<tr><td>{adapter}</td><td>{info.get('percentage', 0):.2f}%</td><td class='{status_class}'>{status_text}</td></tr>")
                f.write("</table>")
            
            # Plots
            f.write("<h3>Visualizations</h3>")
            
            plot_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_length_distribution.png")
            rel_path = os.path.basename(plot_file)
            f.write(f"<div><img src='{rel_path}' alt='Length Distribution'></div>")
            
            quality_plot = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_quality_heatmap.png")
            rel_path = os.path.basename(quality_plot)
            f.write(f"<div><img src='{rel_path}' alt='Quality Heatmap'></div>")
            
            f.write("</div>") # End file div
            
        # End HTML report
        f.write("""
        </body>
        </html>
        """)
    
    logger.info(f"Report generated: {report_file}")
    return report_file

def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Starting QC analysis for {len(args.input)} files")
    
    # Prepare tasks
    tasks = [(file, args) for file in args.input]
    
    # Process files
    results = []
    if args.threads > 1 and len(tasks) > 1:
        with mp.Pool(processes=min(args.threads, len(tasks))) as pool:
            results = pool.map(analyze_fastq, tasks)
    else:
        results = [analyze_fastq(task) for task in tasks]
    
    # Generate plots for each file
    for result in results:
        filename = result["filename"]
        base_name = os.path.splitext(filename)[0]
        
        # Plot length distribution
        length_stats = result.get("length_stats")
        if length_stats:
            plot_file = os.path.join(args.output, f"{base_name}_length_distribution.png")
            plot_length_distribution(length_stats, plot_file)
        
        # Plot quality heatmap
        quality_stats = result.get("quality_stats")
        if quality_stats:
            quality_plot = os.path.join(args.output, f"{base_name}_quality_heatmap.png")
            plot_quality_heatmap(quality_stats, quality_plot)
    
    # Generate report
    report_file = generate_report(results, args.output)
    
    logger.info("QC analysis completed successfully")
    logger.info(f"Results available in {args.output}")
    logger.info(f"Report: {report_file}")

if __name__ == "__main__":
    main()