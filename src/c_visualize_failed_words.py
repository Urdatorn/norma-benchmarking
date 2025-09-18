import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from collections import Counter

from baseline_pos import get_baseline_pos_distribution
from baseline_counts import word_and_syll_baselines

def main():

    input_file = "failed_words/failed_words_with_analysis.tsv"
    pos_tags = []
    failed_words = []
    failed_lemmata = []
    failed_syllables = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pos = row['pos'].strip()
            if pos:  # Only include non-empty POS tags
                pos_tags.append(pos)
            failed_words.append(row['failed_word'])
            
            lemma = row['lemma'].strip()
            if lemma:  # Only include non-empty lemmata
                failed_lemmata.append(lemma)
                
            # Collect failed syllables
            failed_syll = row['failed_syllable'].replace('^', '').replace('_', '').strip()
            if failed_syll:  # Only include non-empty syllables
                failed_syllables.append(failed_syll)

    # Count POS tag frequencies
    pos_counts = Counter(pos_tags)

    # Count failed word frequencies
    word_counts = Counter(failed_words)

    # Count failed lemma frequencies
    lemma_counts = Counter(failed_lemmata)

    # Count failed syllable frequencies
    syllable_counts = Counter(failed_syllables)

    # Get baseline POS distribution
    baseline_pos_counts = get_baseline_pos_distribution()

    # Baselines 
    baseline_word_counts, baseline_syllable_counts = word_and_syll_baselines()

    # Prepare data for grouped bar chart
    # Get all unique POS tags from both failed and baseline data
    all_pos_tags = set(pos_counts.keys()) | set(baseline_pos_counts.keys())
    all_pos_tags = sorted(list(all_pos_tags))  # Sort for consistent ordering
    
    # Get counts for each POS tag, defaulting to 0 if not present
    failed_counts = [pos_counts.get(pos, 0) for pos in all_pos_tags]
    baseline_counts = [baseline_pos_counts.get(pos, 0) for pos in all_pos_tags]
    
    # Calculate totals for percentage conversion
    total_failed = sum(failed_counts)
    total_baseline = sum(baseline_counts)
    
    # Convert to percentages
    failed_percentages = [(count / total_failed) * 100 if total_failed > 0 else 0 for count in failed_counts]
    baseline_percentages = [(count / total_baseline) * 100 if total_baseline > 0 else 0 for count in baseline_counts]

    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    x = np.arange(len(all_pos_tags))  # Label locations
    width = 0.35  # Width of the bars

    # Create the bars
    bars1 = plt.bar(x - width/2, failed_percentages, width, label='Failed Words', color='#ff7f7f', alpha=0.8)
    bars2 = plt.bar(x + width/2, baseline_percentages, width, label='Baseline', color='#7f7fff', alpha=0.8)

    plt.title('Distribution of Part-of-Speech Classes: Failed Words vs Baseline (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Part-of-Speech Tags', fontsize=12)
    plt.ylabel('Percentage of Words', fontsize=12)
    plt.xticks(x, all_pos_tags, rotation=45, ha='right')
    plt.legend()

    # Add value labels on top of bars for failed words
    for bar in bars1:
        height = bar.get_height()
        if height > 0:  # Only add label if there's a value
            plt.text(bar.get_x() + bar.get_width()/2., height + max(failed_percentages) * 0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/pos.png', dpi=300, bbox_inches='tight')
    #plt.show()  # Commented out to only save, not display

    # Print summary statistics
    print("POS Tag Distribution Comparison:")
    print("-" * 60)
    print(f"{'POS Tag':8s} {'Failed':>8s} {'Baseline':>10s} {'Failed %':>10s} {'Baseline %':>12s}")
    print("-" * 60)
    total_failed = sum(pos_counts.values())
    total_baseline = sum(baseline_pos_counts.values())
    
    for pos in all_pos_tags:
        failed_count = pos_counts.get(pos, 0)
        baseline_count = baseline_pos_counts.get(pos, 0)
        failed_pct = (failed_count / total_failed) * 100 if total_failed > 0 else 0
        baseline_pct = (baseline_count / total_baseline) * 100 if total_baseline > 0 else 0
        
        if failed_count > 0 or baseline_count > 0:  # Only show POS tags that appear in at least one dataset
            print(f"{pos:8s} {failed_count:8d} {baseline_count:10d} {failed_pct:9.1f}% {baseline_pct:11.1f}%")
    
    print("-" * 60)
    print(f"{'Total':8s} {total_failed:8d} {total_baseline:10d} {100.0:9.1f}% {100.0:11.1f}%")

    print("\n" + "="*50)
    print("TOP 10 MOST COMMON FAILED WORDS (with baselines):")
    print("="*50)
    for i, (word, count) in enumerate(word_counts.most_common(10), 1):
        baseline_count = baseline_word_counts.get(word, 0)
        print(f"{i:2d}. {word:20s} - {count:3d} failures | baseline: {baseline_count}")
    print("="*50)

    print("\n" + "="*50)
    print("TOP 10 MOST COMMON FAILED LEMMATA:")
    print("="*50)
    for i, (lemma, count) in enumerate(lemma_counts.most_common(10), 1):
        print(f"{i:2d}. {lemma:20s} - {count:3d} occurrences")
    print("="*50)

    print("\n" + "="*50)
    print("TOP 10 MOST COMMON FAILED SYLLABLES (with baselines):")
    print("="*50)
    for i, (syllable, count) in enumerate(syllable_counts.most_common(10), 1):
        percentage = (count / len(failed_syllables)) * 100
        baseline_count = baseline_syllable_counts.get(syllable, 0)
        print(f"{i:2d}. {syllable:15s} - {count:3d} failures ({percentage:.1f}%) | baseline: {baseline_count}")
    print("="*50)

if __name__ == "__main__":
    main()