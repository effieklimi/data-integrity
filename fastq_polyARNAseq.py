#!/usr/bin/env python3
"""
FASTQ Quality Control Script for polyA RNA-seq Data

This script performs comprehensive quality control checks on FASTQ files from
RNA-seq experiments, including:
- Basic file integrity validation
- Read quality metrics
- Adapter content assessment
- GC content distribution
- Sequence complexity analysis
- k-mer overrepresentation
- Potential sample contamination indicators
- polyA tail detection relevant to polyA-selected RNA-seq
"""

import os
import sys
import gzip
import argparse
import statistics
import collections
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import GC
import pandas as pd

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RNA-seq FASTQ Quality Control')
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input FASTQ file(s). Can specify multiple files.')
    parser.add_argument('-o', '--output', default='qc_report',
                        help='Output directory for QC reports')
    parser.add_argument('-a', '--adapters', default=None,
                        help='File containing adapter sequences to check')
    parser.add_argument('-t', '--threads', type=int, default=multiprocessing.cpu_count(),
                        help='Number of threads to use')
    parser.add_argument('-s', '--subsample', type=int, default=1000000,
                        help='Number of reads to subsample for faster analysis')
    parser.add_argument('--min-phred', type=int, default=20,
                        help='Minimum Phred score threshold for quality filtering')
    parser.add_argument('--polya-threshold', type=int, default=10, 
                        help='Minimum length of polyA sequence to detect')
    return parser.parse_args()

def check_file_integrity(filename):
    """Check if FASTQ file is valid and can be properly opened"""
    try:
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rt') as f:
                # Try reading first few lines
                for _ in range(400):
                    if not f.readline():
                        break
        else:
            with open(filename, 'r') as f:
                for _ in range(400):
                    if not f.readline():
                        break
        return True
    except Exception as e:
        print(f"Error checking file integrity of {filename}: {e}")
        return False

def detect_encoding(filename, num_reads=10000):
    """
    Detect quality score encoding (Phred+33 or Phred+64)
    Returns 'Phred+33' (Sanger/Illumina 1.8+) or 'Phred+64' (Illumina 1.3-1.5)
    """
    min_score = 255
    max_score = 0
    
    # Process first num_reads records
    read_count = 0
    try:
        opener = gzip.open if filename.endswith('.gz') else open
        mode = 'rt' if filename.endswith('.gz') else 'r'
        
        with opener(filename, mode) as f:
            while read_count < num_reads:
                # Read 4 lines (one FASTQ record)
                header = f.readline()
                if not header:
                    break
                seq = f.readline()
                f.readline()  # + line
                qual = f.readline().strip()
                
                # Check quality scores
                for char in qual:
                    score = ord(char)
                    min_score = min(min_score, score)
                    max_score = max(max_score, score)
                
                read_count += 1
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return "Unknown"
    
    # Determine encoding based on score ranges
    if min_score >= 33 and max_score <= 74:
        return "Phred+33 (Sanger/Illumina 1.8+)"
    elif min_score >= 64 and max_score <= 104:
        return "Phred+64 (Illumina 1.3-1.5)"
    else:
        return "Mixed or unknown encoding"

def calculate_base_stats(reads):
    """Calculate base composition statistics"""
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
    position_base_counts = {}
    gc_content = []
    read_lengths = []
    
    for record in reads:
        seq = str(record.seq)
        read_lengths.append(len(seq))
        
        # Count bases in this read
        read_bases = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
        for i, base in enumerate(seq):
            if base in base_counts:
                base_counts[base] += 1
                read_bases[base] += 1
            
            # Track base composition by position
            if i not in position_base_counts:
                position_base_counts[i] = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
            if base in position_base_counts[i]:
                position_base_counts[i][base] += 1
        
        # Calculate GC content for this read
        total_bases = sum(read_bases.values())
        if total_bases > 0:
            gc = ((read_bases['G'] + read_bases['C']) / total_bases) * 100
            gc_content.append(gc)
    
    return {
        'base_counts': base_counts,
        'position_base_counts': position_base_counts,
        'gc_content': gc_content,
        'read_lengths': read_lengths
    }

def calculate_quality_stats(reads, encoding='Phred+33'):
    """Calculate quality score statistics"""
    offset = 33 if 'Phred+33' in encoding else 64
    position_qual = {}
    overall_quals = []
    
    for record in reads:
        quals = record.letter_annotations["phred_quality"]
        overall_quals.extend(quals)
        
        for i, qual in enumerate(quals):
            if i not in position_qual:
                position_qual[i] = []
            position_qual[i].append(qual)
    
    # Calculate stats by position
    position_stats = {}
    for pos, quals in position_qual.items():
        position_stats[pos] = {
            'mean': statistics.mean(quals) if quals else 0,
            'median': statistics.median(quals) if quals else 0,
            'q1': np.percentile(quals, 25) if quals else 0,
            'q3': np.percentile(quals, 75) if quals else 0
        }
    
    return {
        'position_stats': position_stats,
        'overall_mean': statistics.mean(overall_quals) if overall_quals else 0,
        'overall_median': statistics.median(overall_quals) if overall_quals else 0
    }

def detect_adapters(reads, adapters):
    """Check for adapter contamination"""
    adapter_counts = {adapter: 0 for adapter in adapters}
    total_reads = 0
    
    for record in reads:
        total_reads += 1
        seq = str(record.seq)
        
        for adapter in adapters:
            if adapter in seq:
                adapter_counts[adapter] += 1
    
    # Calculate percentage
    adapter_percentages = {}
    for adapter, count in adapter_counts.items():
        if total_reads > 0:
            adapter_percentages[adapter] = (count / total_reads) * 100
        else:
            adapter_percentages[adapter] = 0
    
    return adapter_percentages

def calculate_sequence_complexity(reads, kmer_size=5):
    """Analyze sequence complexity and overrepresented kmers"""
    kmer_counts = collections.Counter()
    total_kmers = 0
    
    for record in reads:
        seq = str(record.seq)
        for i in range(len(seq) - kmer_size + 1):
            kmer = seq[i:i+kmer_size]
            if 'N' not in kmer:  # Skip kmers with N's
                kmer_counts[kmer] += 1
                total_kmers += 1
    
    # Calculate top overrepresented kmers
    top_kmers = kmer_counts.most_common(50)
    
    # Calculate expected frequency based on random distribution
    expected_freq = 1 / (4 ** kmer_size)
    
    overrepresented_kmers = []
    for kmer, count in top_kmers:
        observed_freq = count / total_kmers if total_kmers > 0 else 0
        fold_enrichment = observed_freq / expected_freq if expected_freq > 0 else 0
        
        if fold_enrichment > 10:  # 10x more than expected by chance
            overrepresented_kmers.append({
                'kmer': kmer,
                'count': count,
                'percentage': observed_freq * 100,
                'fold_enrichment': fold_enrichment
            })
    
    return overrepresented_kmers

def detect_polya_tails(reads, min_length=10):
    """
    Detect potential polyA tails in reads
    Specific to polyA RNA-seq experiments
    """
    polya_counts = 0
    polyt_counts = 0
    total_reads = 0
    tail_length_dist = []
    
    for record in reads:
        total_reads += 1
        seq = str(record.seq)
        
        # Look for polyA at the end of reads (typical for 3' end RNA-seq)
        max_a_streak = 0
        for i in range(len(seq) - min_length + 1):
            if seq[i:i+min_length] == 'A' * min_length:
                # Count consecutive As
                j = i + min_length
                while j < len(seq) and seq[j] == 'A':
                    j += 1
                streak_len = j - i
                max_a_streak = max(max_a_streak, streak_len)
                
        # Look for polyT at the beginning (might indicate reverse complement)
        max_t_streak = 0
        for i in range(min(len(seq), 30)):  # Only check first 30bp
            if i + min_length <= len(seq) and seq[i:i+min_length] == 'T' * min_length:
                # Count consecutive Ts
                j = i + min_length
                while j < len(seq) and seq[j] == 'T':
                    j += 1
                streak_len = j - i
                max_t_streak = max(max_t_streak, streak_len)
        
        if max_a_streak >= min_length:
            polya_counts += 1
            tail_length_dist.append(max_a_streak)
        
        if max_t_streak >= min_length:
            polyt_counts += 1
            
    return {
        'polya_percentage': (polya_counts / total_reads) * 100 if total_reads > 0 else 0,
        'polyt_percentage': (polyt_counts / total_reads) * 100 if total_reads > 0 else 0,
        'tail_length_distribution': collections.Counter(tail_length_dist)
    }

def check_read_duplicates(reads, sample_size=50000):
    """Check for PCR duplicates by comparing first N bases of reads"""
    start_sequences = collections.Counter()
    sampled_reads = 0
    
    for record in reads:
        if sampled_reads >= sample_size:
            break
            
        # Use first 50bp as signature (or full read if shorter)
        start_seq = str(record.seq)[:50]
        start_sequences[start_seq] += 1
        sampled_reads += 1
    
    # Calculate duplication metrics
    total_sampled = len(start_sequences)
    if total_sampled == 0:
        return {'duplicate_rate': 0, 'unique_starts': 0}
    
    unique_starts = sum(1 for count in start_sequences.values() if count == 1)
    duplicate_rate = 1 - (unique_starts / sampled_reads)
    
    return {
        'duplicate_rate': duplicate_rate * 100,
        'unique_starts': unique_starts,
        'total_sampled': sampled_reads
    }

def create_plots(stats, filename, output_dir):
    """Create QC plots and save to output directory"""
    base_filename = os.path.basename(filename).split('.')[0]
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Per-base quality plot
    plt.figure(figsize=(12, 6))
    positions = sorted(stats['quality_stats']['position_stats'].keys())
    means = [stats['quality_stats']['position_stats'][p]['mean'] for p in positions]
    q1s = [stats['quality_stats']['position_stats'][p]['q1'] for p in positions]
    q3s = [stats['quality_stats']['position_stats'][p]['q3'] for p in positions]
    
    plt.plot(positions, means, 'b-', label='Mean Quality')
    plt.fill_between(positions, q1s, q3s, alpha=0.2, color='b', label='25-75 percentile')
    plt.axhline(y=20, color='r', linestyle='--', label='Q20 threshold')
    plt.axhline(y=30, color='g', linestyle='--', label='Q30 threshold')
    
    plt.xlabel('Position in read (bp)')
    plt.ylabel('Quality score')
    plt.title(f'Per-base quality scores - {base_filename}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_base_quality.png'))
    plt.close()
    
    # 2. GC content distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats['base_stats']['gc_content'], bins=50, alpha=0.75, edgecolor='black')
    plt.axvline(x=statistics.mean(stats['base_stats']['gc_content']), 
                color='r', linestyle='--', 
                label=f'Mean GC: {statistics.mean(stats["base_stats"]["gc_content"]):.2f}%')
    
    plt.xlabel('GC Content (%)')
    plt.ylabel('Frequency')
    plt.title(f'GC Content Distribution - {base_filename}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_gc_content.png'))
    plt.close()
    
    # 3. Per-base sequence content
    plt.figure(figsize=(12, 6))
    positions = sorted(stats['base_stats']['position_base_counts'].keys())
    
    # Ensure we only plot a reasonable number of positions
    if len(positions) > 100:
        # Sample positions if there are too many
        step = len(positions) // 100
        positions = positions[::step] + [positions[-1]]
    
    for base in ['A', 'C', 'G', 'T']:
        base_pcts = []
        for pos in positions:
            pos_counts = stats['base_stats']['position_base_counts'][pos]
            total = sum(pos_counts.values())
            pct = (pos_counts[base] / total * 100) if total > 0 else 0
            base_pcts.append(pct)
        
        plt.plot(positions, base_pcts, label=base)
    
    plt.xlabel('Position in read (bp)')
    plt.ylabel('Percentage')
    plt.title(f'Per-base sequence content - {base_filename}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_base_content.png'))
    plt.close()
    
    # 4. Read length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats['base_stats']['read_lengths'], bins=50, alpha=0.75, edgecolor='black')
    
    plt.xlabel('Read Length (bp)')
    plt.ylabel('Frequency')
    plt.title(f'Read Length Distribution - {base_filename}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_read_length.png'))
    plt.close()
    
    # 5. PolyA tail length distribution if available
    if 'polya_stats' in stats and stats['polya_stats']['tail_length_distribution']:
        plt.figure(figsize=(10, 6))
        lengths = list(stats['polya_stats']['tail_length_distribution'].keys())
        counts = list(stats['polya_stats']['tail_length_distribution'].values())
        
        plt.bar(lengths, counts)
        plt.xlabel('PolyA Tail Length (bp)')
        plt.ylabel('Frequency')
        plt.title(f'PolyA Tail Length Distribution - {base_filename}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_filename}_polya_length.png'))
        plt.close()
    
    return True

def generate_report(stats, filename, output_dir):
    """Generate HTML and text report with QC results"""
    base_filename = os.path.basename(filename).split('.')[0]
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create text report
    report_file = os.path.join(output_dir, f'{base_filename}_qc_report.txt')
    
    with open(report_file, 'w') as f:
        f.write(f"======= FASTQ QC Report: {filename} =======\n\n")
        
        # Basic stats
        f.write("=== Basic Statistics ===\n")
        f.write(f"File: {filename}\n")
        f.write(f"Encoding: {stats['encoding']}\n")
        
        # Calculate a few more metrics
        total_bases = sum(stats['base_stats']['base_counts'].values())
        read_count = len(stats['base_stats']['read_lengths'])
        avg_read_len = statistics.mean(stats['base_stats']['read_lengths']) if stats['base_stats']['read_lengths'] else 0
        
        f.write(f"Total reads: {read_count:,}\n")
        f.write(f"Total bases: {total_bases:,}\n")
        f.write(f"Average read length: {avg_read_len:.2f} bp\n")
        f.write(f"Average quality score: {stats['quality_stats']['overall_mean']:.2f}\n\n")
        
        # Base composition
        f.write("=== Base Composition ===\n")
        for base, count in stats['base_stats']['base_counts'].items():
            percentage = (count / total_bases * 100) if total_bases > 0 else 0
            f.write(f"{base}: {count:,} ({percentage:.2f}%)\n")
        
        avg_gc = statistics.mean(stats['base_stats']['gc_content']) if stats['base_stats']['gc_content'] else 0
        f.write(f"Average GC content: {avg_gc:.2f}%\n\n")
        
        # Quality scores
        f.write("=== Quality Scores ===\n")
        f.write(f"Mean quality: {stats['quality_stats']['overall_mean']:.2f}\n")
        f.write(f"Median quality: {stats['quality_stats']['overall_median']:.2f}\n\n")
        
        # Duplication rates
        f.write("=== Duplication Rates ===\n")
        f.write(f"Estimated duplication rate: {stats['duplicate_stats']['duplicate_rate']:.2f}%\n")
        f.write(f"Unique start sequences: {stats['duplicate_stats']['unique_starts']:,} ")
        f.write(f"(out of {stats['duplicate_stats']['total_sampled']:,} sampled)\n\n")
        
        # Overrepresented sequences
        f.write("=== Overrepresented k-mers ===\n")
        if len(stats['complexity_stats']) > 0:
            for i, kmer_info in enumerate(stats['complexity_stats'][:10]):  # Show top 10
                f.write(f"{i+1}. {kmer_info['kmer']}: {kmer_info['count']:,} occurrences ")
                f.write(f"({kmer_info['percentage']:.2f}%, {kmer_info['fold_enrichment']:.1f}x enriched)\n")
        else:
            f.write("No significantly overrepresented k-mers detected\n")
        f.write("\n")
        
        # Adapter content
        if 'adapter_stats' in stats:
            f.write("=== Adapter Content ===\n")
            for adapter, percentage in stats['adapter_stats'].items():
                f.write(f"{adapter}: {percentage:.2f}% of reads\n")
            f.write("\n")
        
        # PolyA content (specific to RNA-seq)
        if 'polya_stats' in stats:
            f.write("=== PolyA Content ===\n")
            f.write(f"Reads with polyA tail: {stats['polya_stats']['polya_percentage']:.2f}%\n")
            f.write(f"Reads with polyT at start: {stats['polya_stats']['polyt_percentage']:.2f}%\n\n")
        
        # Warnings and recommendations
        f.write("=== Quality Assessment ===\n")
        warnings = []
        
        # Check for quality issues
        if stats['quality_stats']['overall_mean'] < 20:
            warnings.append("Low average quality score (<20) indicates potential sequencing problems")
        
        # Check for adapter content
        if 'adapter_stats' in stats:
            high_adapter = any(pct > 10 for pct in stats['adapter_stats'].values())
            if high_adapter:
                warnings.append("High adapter content (>10%) detected - consider trimming")
        
        # Check for GC bias
        avg_gc = statistics.mean(stats['base_stats']['gc_content']) if stats['base_stats']['gc_content'] else 0
        if avg_gc < 35 or avg_gc > 65:
            warnings.append(f"Unusual GC content ({avg_gc:.1f}%) - check for contamination or bias")
        
        # Check for duplication
        if stats['duplicate_stats']['duplicate_rate'] > 50:
            warnings.append("High duplication rate (>50%) suggests PCR artifacts or low complexity library")
        
        # Check for polyA/T balance in RNA-seq
        if 'polya_stats' in stats:
            polyt_high = stats['polya_stats']['polyt_percentage'] > 10
            if polyt_high:
                warnings.append("High polyT content at read starts suggests potential strand issues in library prep")
        
        # Output warnings
        if warnings:
            f.write("WARNINGS:\n")
            for warning in warnings:
                f.write(f"- {warning}\n")
        else:
            f.write("No significant quality issues detected.\n")
    
    # Return path to report
    return report_file

def process_fastq(filename, args):
    """Process a single FASTQ file and generate QC metrics"""
    print(f"Processing {filename}...")
    
    # Check file integrity
    if not check_file_integrity(filename):
        print(f"ERROR: File {filename} appears to be corrupt or invalid")
        return None
    
    # Detect quality encoding
    encoding = detect_encoding(filename)
    print(f"Detected quality encoding: {encoding}")
    
    # Load adapters if provided
    adapters = []
    if args.adapters:
        try:
            with open(args.adapters, 'r') as f:
                adapters = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading adapter file: {e}")
    
    # Use common Illumina adapters if not provided
    if not adapters:
        adapters = [
            "AGATCGGAAGAGC",  # Standard Illumina adapter
            "CTGTCTCTTATACACATCT"  # Nextera adapter
        ]
    
    # Load a subsample of reads for analysis
    reads = []
    try:
        format_type = "fastq-illumina" if "Phred+64" in encoding else "fastq-sanger"
        opener = gzip.open if filename.endswith('.gz') else open
        mode = 'rt' if filename.endswith('.gz') else 'r'
        
        with opener(filename, mode) as handle:
            for i, record in enumerate(SeqIO.parse(handle, format_type)):
                reads.append(record)
                if i + 1 >= args.subsample:
                    break
    except Exception as e:
        print(f"Error reading FASTQ file: {e}")
        return None
    
    print(f"Loaded {len(reads)} reads for analysis")
    
    # Calculate statistics
    stats = {}
    stats['encoding'] = encoding
    
    # Base composition
    print("Calculating base statistics...")
    stats['base_stats'] = calculate_base_stats(reads)
    
    # Quality stats
    print("Calculating quality statistics...")
    stats['quality_stats'] = calculate_quality_stats(reads, encoding)
    
    # Adapter content
    print("Checking adapter content...")
    stats['adapter_stats'] = detect_adapters(reads, adapters)
    
    # Sequence complexity and overrepresentation
    print("Analyzing sequence complexity...")
    stats['complexity_stats'] = calculate_sequence_complexity(reads)
    
    # Duplication rate
    print("Checking for duplicates...")
    stats['duplicate_stats'] = check_read_duplicates(reads)
    
    # PolyA content (specific to RNA-seq)
    print("Detecting polyA tails...")
    stats['polya_stats'] = detect_polya_tails(reads, args.polya_threshold)
    
    # Generate plots
    print("Creating plots...")
    create_plots(stats, filename, args.output)
    
    # Generate report
    print("Generating report...")
    report_file = generate_report(stats, filename, args.output)
    
    print(f"QC analysis complete for {filename}")
    print(f"Report saved to {report_file}")
    
    return stats

def main():
    """Main function to process FASTQ files"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each input file
    results = {}
    for filename in args.input:
        if not os.path.exists(filename):
            print(f"ERROR: File {filename} not found")
            continue
        
        stats = process_fastq(filename, args)
        if stats:
            results[filename] = stats
    
    # Generate a summary report if multiple files were processed
    if len(results) > 1:
        summary_file = os.path.join(args.output, "summary_report.txt")
        
        with open(summary_file, 'w') as f:
            f.write("======= FASTQ QC Summary Report =======\n\n")
            
            for filename, stats in results.items():
                base_filename = os.path.basename(filename)
                f.write(f"File: {base_filename}\n")
                
                # Calculate key metrics
                total_bases = sum(stats['base_stats']['base_counts'].values())
                read_count = len(stats['base_stats']['read_lengths'])
                avg_read_len = statistics.mean(stats['base_stats']['read_lengths']) if stats['base_stats']['read_lengths'] else 0
                avg_gc = statistics.mean(stats['base_stats']['gc_content']) if stats['base_stats']['gc_content'] else 0
                
                f.write(f"  Reads: {read_count:,}\n")
                f.write(f"  Avg Length: {avg_read_len:.2f} bp\n")
                f.write(f"  Avg Quality: {stats['quality_stats']['overall_mean']:.2f}\n")
                f.write(f"  GC Content: {avg_gc:.2f}%\n")
                f.write(f"  Duplication: {stats['duplicate_stats']['duplicate_rate']:.2f}%\n")
                
                if 'polya_stats' in stats:
                    f.write(f"  PolyA Content: {stats['polya_stats']['polya_percentage']:.2f}%\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {summary_file}")
    
    print("All QC analyses complete")

if __name__ == "__main__":
    main()