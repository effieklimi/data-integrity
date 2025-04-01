#!/usr/bin/env python3
"""
sc_rnaseq_qc.py - Quality Control for Single-Cell RNA-Seq FASTQ files

This script performs robust data integrity and quality control checks on FASTQ files
from single-cell RNA sequencing experiments, including:
- File format validation
- Cell barcode and UMI quality assessment
- Read quality metrics
- Barcode distribution analysis
- Cell multiplet detection
- Ambient RNA estimation
- Basic summary statistics for cell and gene counts

Usage:
    python sc_rnaseq_qc.py -1 R1.fastq -2 R2.fastq [-o output_directory] [--protocol 10x] [options]

Requirements:
    - Python 3.6+
    - Biopython
    - Matplotlib
    - Pandas
    - NumPy
    - Scikit-learn (for clustering analysis)
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
import itertools
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from Bio import SeqIO
from Bio.SeqUtils import GC
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common scRNA-seq protocol configurations
PROTOCOLS = {
    "10x-v2": {
        "description": "10x Genomics Chromium v2",
        "r1_cell_barcode_pos": (0, 16),  # Cell barcode is first 16bp in R1
        "r1_umi_pos": (16, 26),          # UMI is next 10bp in R1
        "whitelist_file": "10x_v2_barcodes.txt",  # File with valid cell barcodes
        "expected_cells": 10000,
    },
    "10x-v3": {
        "description": "10x Genomics Chromium v3",
        "r1_cell_barcode_pos": (0, 16),  # Cell barcode is first 16bp in R1
        "r1_umi_pos": (16, 28),          # UMI is next 12bp in R1
        "whitelist_file": "10x_v3_barcodes.txt",
        "expected_cells": 10000,
    },
    "drop-seq": {
        "description": "Drop-seq",
        "r1_cell_barcode_pos": (0, 12),  # Cell barcode is first 12bp in R1
        "r1_umi_pos": (12, 20),          # UMI is next 8bp in R1
        "whitelist_file": None,          # No standard whitelist
        "expected_cells": 5000,
    },
    "indrops": {
        "description": "inDrop",
        "r1_cell_barcode_pos": (0, 11),  # Cell barcode construction is more complex
        "r1_umi_pos": (11, 19),          # UMI length can vary
        "whitelist_file": None,
        "expected_cells": 5000,
    },
    "smart-seq2": {
        "description": "Smart-seq2 (full-length)",
        "r1_cell_barcode_pos": None,     # No cell barcode in reads (plate-based)
        "r1_umi_pos": None,              # Typically no UMIs
        "whitelist_file": None,
        "expected_cells": 96,            # Typically done in plates
    },
    "cel-seq2": {
        "description": "CEL-Seq2",
        "r1_cell_barcode_pos": (0, 6),   # Cell barcode is first 6bp in R1
        "r1_umi_pos": (6, 12),           # UMI is next 6bp in R1
        "whitelist_file": None,
        "expected_cells": 96,
    },
    "custom": {
        "description": "Custom protocol (configure manually)",
        "r1_cell_barcode_pos": (0, 16),  # Default values
        "r1_umi_pos": (16, 26),
        "whitelist_file": None,
        "expected_cells": 5000,
    }
}

# Common sequencing adapters for RNA-seq
ADAPTERS = {
    "Illumina_TruSeq": "AGATCGGAAGAGC",
    "Nextera": "CTGTCTCTTATACACATCT",
    "10x_TSO": "AAGCAGTGGTATCAACGCAGAGTACATGGG",
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quality control for single-cell RNA-seq FASTQ files")
    
    # Input files
    parser.add_argument("-1", "--read1", required=True, help="R1 FASTQ file(s) (cell barcode/UMI)", nargs='+')
    parser.add_argument("-2", "--read2", required=True, help="R2 FASTQ file(s) (cDNA)", nargs='+')
    parser.add_argument("-i", "--index", help="Index FASTQ file(s) (if separate)", nargs='+')
    
    # Output options
    parser.add_argument("-o", "--output", default="./scqc_output", help="Output directory")
    
    # Protocol selection
    parser.add_argument("--protocol", choices=list(PROTOCOLS.keys()), default="10x-v3",
                        help="Single-cell protocol")
    
    # Custom protocol settings
    parser.add_argument("--barcode-pos", type=lambda x: tuple(map(int, x.split(','))),
                        help="Cell barcode position in R1 (start,end)")
    parser.add_argument("--umi-pos", type=lambda x: tuple(map(int, x.split(','))),
                        help="UMI position in R1 (start,end)")
    parser.add_argument("--expected-cells", type=int, 
                        help="Expected number of cells")
    parser.add_argument("--whitelist", help="Cell barcode whitelist file")
    
    # QC options
    parser.add_argument("-t", "--threads", type=int, default=mp.cpu_count(), 
                        help="Number of threads to use")
    parser.add_argument("--sample-size", type=int, default=1000000, 
                        help="Number of reads to sample (0 for all)")
    parser.add_argument("--phred-encoding", choices=["phred33", "phred64"], default="phred33", 
                        help="FASTQ quality score encoding")
    parser.add_argument("--min-counts-per-cell", type=int, default=500,
                        help="Minimum counts per cell")
    parser.add_argument("--min-genes-per-cell", type=int, default=200,
                        help="Minimum genes per cell")
    parser.add_argument("--max-mito-pct", type=float, default=20.0,
                        help="Maximum mitochondrial percentage")
    parser.add_argument("--knee-method", choices=["inflection", "dropoff"], default="inflection",
                        help="Method to detect cell number from barcode knee plot")
    
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
        return True, "File appears to be valid FASTQ"
    except Exception as e:
        return False, f"Error checking file integrity: {str(e)}"

def count_reads(fastq_file):
    """Count the total number of reads in a FASTQ file."""
    try:
        # For large files, estimate by counting lines and dividing by 4
        if os.path.getsize(fastq_file) > 1e9:  # > 1GB
            line_count = 0
            with open_fastq(fastq_file) as f:
                for i, _ in enumerate(f):
                    line_count += 1
                    if i >= 400000:  # Count first 100K records
                        break
            if line_count < 4:
                return 0
            return line_count // 4
        else:
            count = 0
            with open_fastq(fastq_file) as f:
                for _ in SeqIO.parse(f, "fastq"):
                    count += 1
            return count
    except Exception as e:
        logger.error(f"Error counting reads: {str(e)}")
        return 0

def calculate_phred_scores(record, phred_encoding):
    """Calculate quality scores from FASTQ quality string."""
    if phred_encoding == "phred33":
        offset = 33
    else:  # phred64
        offset = 64
    
    quality_scores = [ord(q) - offset for q in record.letter_annotations["phred_quality"]]
    return quality_scores

def load_barcode_whitelist(whitelist_file):
    """Load cell barcode whitelist if provided."""
    if not whitelist_file or not os.path.exists(whitelist_file):
        return None
    
    try:
        whitelist = set()
        with open(whitelist_file, 'r') as f:
            for line in f:
                whitelist.add(line.strip())
        return whitelist
    except Exception as e:
        logger.error(f"Error loading whitelist: {str(e)}")
        return None

def hamming_distance(s1, s2):
    """Calculate Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        return float('inf')
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def correct_barcode(barcode, whitelist, max_mismatches=1):
    """Correct a barcode using a whitelist, if within Hamming distance threshold."""
    if barcode in whitelist:
        return barcode, 0
    
    # Check for barcodes within max_mismatches
    for valid_barcode in whitelist:
        distance = hamming_distance(barcode, valid_barcode)
        if distance <= max_mismatches:
            return valid_barcode, distance
    
    return barcode, -1  # No match found

def extract_cell_barcode_umi(record, protocol, whitelist=None):
    """Extract cell barcode and UMI from a read based on protocol settings."""
    sequence = str(record.seq)
    quality = record.letter_annotations["phred_quality"]
    
    barcode_pos = protocol["r1_cell_barcode_pos"]
    umi_pos = protocol["r1_umi_pos"]
    
    if barcode_pos is None or umi_pos is None:
        # Protocol doesn't use barcode/UMI in reads
        return None, None, None, None
    
    # Extract barcode and UMI
    barcode = sequence[barcode_pos[0]:barcode_pos[1]] if len(sequence) >= barcode_pos[1] else None
    umi = sequence[umi_pos[0]:umi_pos[1]] if len(sequence) >= umi_pos[1] else None
    
    # Extract quality scores
    barcode_qual = quality[barcode_pos[0]:barcode_pos[1]] if len(quality) >= barcode_pos[1] else None
    umi_qual = quality[umi_pos[0]:umi_pos[1]] if len(quality) >= umi_pos[1] else None
    
    # Correct barcode if whitelist provided
    corrected_barcode = barcode
    if whitelist and barcode and barcode not in whitelist:
        corrected_barcode, _ = correct_barcode(barcode, whitelist)
    
    return barcode, corrected_barcode, umi, (barcode_qual, umi_qual)

def detect_knee_point(counts, method="inflection"):
    """
    Detect the knee point in the barcode count distribution.
    This is used to estimate the number of real cells vs. empty droplets.
    """
    x = np.arange(len(counts))
    y = np.array(counts)
    
    if method == "inflection":
        # Use numerical differentiation to find inflection point
        # First smooth the data
        window = min(101, len(y) // 10)
        if window % 2 == 0:
            window += 1
        
        from scipy.signal import savgol_filter
        y_smooth = savgol_filter(y, window, 3)
        
        # Calculate first and second derivatives
        dy = np.gradient(y_smooth)
        d2y = np.gradient(dy)
        
        # Find where second derivative is approximately zero (inflection point)
        inflection_pts = np.where(np.abs(d2y) < np.std(d2y) * 0.1)[0]
        
        if len(inflection_pts) > 0:
            # Find point with max curvature among inflection candidates
            curvature = np.abs(dy[inflection_pts])
            knee_idx = inflection_pts[np.argmax(curvature)]
            return knee_idx
        else:
            # Fallback to steepest dropoff
            return np.argmax(np.abs(dy))
    else:  # dropoff method
        # Find the point with steepest dropoff
        dropoff = y[:-1] - y[1:]
        return np.argmax(dropoff) + 1

def analyze_cell_barcodes(barcode_counts, protocol, method="inflection"):
    """Analyze cell barcode distribution and detect real cells vs. background."""
    # Sort barcodes by count (descending)
    sorted_barcodes = sorted(barcode_counts.items(), key=lambda x: x[1], reverse=True)
    barcodes = [b for b, _ in sorted_barcodes]
    counts = [c for _, c in sorted_barcodes]
    
    # Get expected number of cells
    expected_cells = protocol["expected_cells"]
    
    # Find knee/inflection point in log curve
    if len(counts) > 0:
        log_counts = np.log10(np.array(counts) + 1)
        knee_idx = detect_knee_point(log_counts, method)
        detected_cells = knee_idx + 1
    else:
        detected_cells = 0
        knee_idx = 0
    
    # Calculate cell stats
    results = {
        "total_barcodes": len(barcodes),
        "expected_cells": expected_cells,
        "detected_cells": detected_cells,
        "barcodes_with_count_>=_10": sum(1 for c in counts if c >= 10),
        "barcodes": barcodes[:min(100000, len(barcodes))],  # Limit to 100K barcodes
        "counts": counts[:min(100000, len(counts))],
        "knee_index": knee_idx,
    }
    
    # Check agreement with expected cells
    if expected_cells > 0:
        ratio = detected_cells / expected_cells
        results["cells_ratio"] = ratio
        results["cells_warning"] = ratio < 0.5 or ratio > 2.0
    
    return results

def calculate_base_stats(sequences):
    """Calculate base composition statistics."""
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
    total_bases = 0
    
    for seq in sequences:
        for base in seq.upper():
            if base in base_counts:
                base_counts[base] += 1
            else:
                base_counts['N'] += 1
            total_bases += 1
    
    # Calculate percentages
    base_pct = {}
    for base, count in base_counts.items():
        base_pct[base] = (count / total_bases) * 100 if total_bases > 0 else 0
    
    return base_counts, base_pct

def estimate_ambient_rna(barcode_counts, cell_barcode_result):
    """
    Estimate ambient RNA contamination level by looking at counts in 
    empty droplets vs. cells.
    """
    # Get cell barcodes
    cell_cutoff = cell_barcode_result["knee_index"]
    barcodes = cell_barcode_result["barcodes"]
    
    if len(barcodes) <= cell_cutoff or cell_cutoff == 0:
        return {
            "ambient_estimation": "Unable to estimate (insufficient data)"
        }
    
    # Cell barcodes
    cell_barcodes = set(barcodes[:cell_cutoff])
    
    # Empty droplets (take a sample of empty droplets)
    empty_start = cell_cutoff + 100  # Skip the boundary cases
    empty_end = min(len(barcodes), empty_start + 10000)
    if empty_end <= empty_start:
        return {
            "ambient_estimation": "Unable to estimate (insufficient empty droplets)"
        }
    
    empty_barcodes = set(barcodes[empty_start:empty_end])
    
    # Calculate metrics
    cell_mean = np.mean([barcode_counts[bc] for bc in cell_barcodes])
    empty_mean = np.mean([barcode_counts[bc] for bc in empty_barcodes])
    
    # Estimate ambient RNA percentage
    if cell_mean > 0:
        ambient_pct = (empty_mean / cell_mean) * 100
    else:
        ambient_pct = 0
    
    return {
        "ambient_pct": ambient_pct,
        "cells_mean_counts": cell_mean,
        "empty_mean_counts": empty_mean,
        "ambient_warning": ambient_pct > 10.0,  # Warning if ambient > 10%
        "ambient_estimation": "High" if ambient_pct > 10 else "Moderate" if ambient_pct > 5 else "Low",
    }

def detect_potential_multiplets(barcode_counts, cell_barcode_result):
    """
    Detect potential cell multiplets based on UMI counts.
    This is a simple approach - more sophisticated methods require gene expression data.
    """
    # Get cell barcodes
    cell_cutoff = cell_barcode_result["knee_index"]
    barcodes = cell_barcode_result["barcodes"]
    
    if len(barcodes) <= cell_cutoff or cell_cutoff == 0:
        return {
            "multiplet_estimation": "Unable to estimate (insufficient data)"
        }
    
    # Get counts for cells
    cell_barcodes = barcodes[:cell_cutoff]
    cell_counts = [barcode_counts[bc] for bc in cell_barcodes]
    
    # Simple approach: cells with counts > 2x median might be multiplets
    median_count = np.median(cell_counts)
    multiplet_threshold = 2 * median_count
    
    potential_multiplets = sum(1 for count in cell_counts if count > multiplet_threshold)
    multiplet_rate = (potential_multiplets / len(cell_counts)) * 100 if len(cell_counts) > 0 else 0
    
    # Expected doublet rate based on cell loading
    expected_cells = cell_barcode_result["expected_cells"]
    if expected_cells > 0:
        # Simple Poisson model for doublet rate
        cells_per_droplet = expected_cells / 10000  # Assuming ~10K droplets
        expected_doublet_rate = 100 * (1 - np.exp(-cells_per_droplet))
    else:
        expected_doublet_rate = None
    
    return {
        "potential_multiplets": potential_multiplets,
        "multiplet_rate": multiplet_rate,
        "multiplet_threshold": multiplet_threshold,
        "expected_doublet_rate": expected_doublet_rate,
        "multiplet_warning": multiplet_rate > 20.0 if expected_doublet_rate is None else multiplet_rate > (expected_doublet_rate * 1.5),
        "multiplet_estimation": "High" if multiplet_rate > 20 else "Moderate" if multiplet_rate > 10 else "Low",
    }

def analyze_umi_saturation(umi_counts_per_cell):
    """
    Analyze UMI saturation to determine sequencing depth adequacy.
    """
    if not umi_counts_per_cell:
        return {
            "saturation_estimation": "Unable to estimate (insufficient data)"
        }
    
    # Calculate UMI saturation by subsampling
    sampled_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    saturation_curve = []
    
    # Get a random selection of cells
    selected_cells = random.sample(
        list(umi_counts_per_cell.keys()),
        min(100, len(umi_counts_per_cell))
    )
    
    # Calculate saturation for selected cells
    for cell in selected_cells:
        umis = umi_counts_per_cell[cell]
        unique_umis = len(set(umis))
        
        if unique_umis == 0:
            continue
            
        cell_saturation = []
        
        for fraction in sampled_fractions:
            sample_size = int(len(umis) * fraction)
            if sample_size == 0:
                cell_saturation.append(0)
                continue
                
            sampled_umis = random.sample(umis, sample_size)
            unique_sampled = len(set(sampled_umis))
            saturation = unique_sampled / unique_umis
            cell_saturation.append(saturation)
            
        saturation_curve.append(cell_saturation)
    
    if not saturation_curve:
        return {
            "saturation_estimation": "Unable to estimate (insufficient data)"
        }
        
    # Average across cells
    avg_saturation = np.mean(saturation_curve, axis=0)
    
    # Calculate gradient at end point - if > 0.01, depth might be insufficient
    gradient = avg_saturation[-1] - avg_saturation[-2]
    
    # Final saturation level
    final_saturation = avg_saturation[-1]
    
    return {
        "saturation_curve": {
            "fractions": sampled_fractions,
            "saturation": avg_saturation.tolist(),
        },
        "final_saturation": final_saturation,
        "saturation_gradient": gradient,
        "saturation_warning": gradient > 0.01,
        "saturation_estimation": "Low" if final_saturation < 0.7 else "Moderate" if final_saturation < 0.9 else "High",
    }

def plot_barcode_knee(cell_data, output_file):
    """Plot barcode count distribution with knee point."""
    barcodes = cell_data["barcodes"]
    counts = cell_data["counts"]
    knee_idx = cell_data["knee_index"]
    expected_cells = cell_data["expected_cells"]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(counts) + 1), counts, '-', linewidth=2)
    
    # Mark knee point
    if knee_idx > 0 and knee_idx < len(counts):
        plt.axvline(x=knee_idx, linestyle='--', color='red', label=f'Detected cells: {knee_idx}')
    
    # Mark expected cells
    if expected_cells > 0:
        plt.axvline(x=expected_cells, linestyle='--', color='green', label=f'Expected cells: {expected_cells}')
    
    plt.xlabel('Barcodes (ranked)')
    plt.ylabel('UMI counts')
    plt.title('Cell Barcode Distribution (Knee Plot)')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_quality_heatmap(quality_data, output_file):
    """Plot quality scores as a heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot R1 (barcode/UMI) quality
    r1_per_position = quality_data["r1"]["per_position_quality"]
    r1_positions = sorted(r1_per_position.keys())
    r1_qualities = [r1_per_position[pos] for pos in r1_positions]
    
    # Plot R1 quality bars
    ax1.bar(r1_positions, r1_qualities, color=[quality_to_color(q) for q in r1_qualities], width=1.0, edgecolor='none')
    
    # Add guidelines
    ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7)
    ax1.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    
    ax1.set_title('R1 Quality Scores (Barcode/UMI)')
    ax1.set_xlabel('Position in Read (bp)')
    ax1.set_ylabel('Mean Quality Score')
    ax1.set_ylim(0, max(40, max(r1_qualities) + 5))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot R2 (cDNA) quality
    r2_per_position = quality_data["r2"]["per_position_quality"]
    r2_positions = sorted(r2_per_position.keys())
    r2_qualities = [r2_per_position[pos] for pos in r2_positions]
    
    # Plot R2 quality bars
    ax2.bar(r2_positions, r2_qualities, color=[quality_to_color(q) for q in r2_qualities], width=1.0, edgecolor='none')
    
    # Add guidelines
    ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    
    ax2.set_title('R2 Quality Scores (cDNA)')
    ax2.set_xlabel('Position in Read (bp)')
    ax2.set_ylabel('Mean Quality Score')
    ax2.set_ylim(0, max(40, max(r2_qualities) + 5))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add color legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color='orange', alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7)
    ]
    labels = ['Poor quality (<20)', 'Reasonable quality (20-30)', 'Good quality (>30)']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_file)
    plt.close()

def plot_barcode_quality(barcode_qual_data, output_file):
    """Plot barcode and UMI quality distribution."""
    barcode_qual = barcode_qual_data.get("barcode_avg_qual", [])
    umi_qual = barcode_qual_data.get("umi_avg_qual", [])
    
    plt.figure(figsize=(10, 6))
    
    # Plot barcode quality distribution
    if barcode_qual:
        plt.hist(barcode_qual, bins=30, alpha=0.7, label='Cell Barcode', color='blue')
    
    # Plot UMI quality distribution
    if umi_qual:
        plt.hist(umi_qual, bins=30, alpha=0.7, label='UMI', color='green')
    
    plt.xlabel('Average Quality Score')
    plt.ylabel('Frequency')
    plt.title('Cell Barcode and UMI Quality Distribution')
    plt.axvline(x=30, linestyle='--', color='red', label='Quality threshold (Q30)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_saturation_curve(saturation_data, output_file):
    """Plot UMI saturation curve."""
    saturation_curve = saturation_data.get("saturation_curve", {})
    fractions = saturation_curve.get("fractions", [])
    saturation = saturation_curve.get("saturation", [])
    
    if not fractions or not saturation or len(fractions) != len(saturation):
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(fractions, saturation, 'o-', linewidth=2, color='blue')
    
    plt.xlabel('Sequencing Depth Fraction')
    plt.ylabel('UMI Saturation')
    plt.title('Sequencing Saturation Curve')
    
    # Add reference lines
    plt.axhline(y=0.8, linestyle='--', color='orange', label='80% Saturation')
    plt.axhline(y=0.9, linestyle='--', color='green', label='90% Saturation')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
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
    
    