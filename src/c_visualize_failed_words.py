import csv
import matplotlib.pyplot as plt
from collections import Counter

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

    # Baselines 
    baseline_word_counts, baseline_syllable_counts = word_and_syll_baselines()

    # Create bar chart
    plt.figure(figsize=(12, 8))
    labels = list(pos_counts.keys())
    sizes = list(pos_counts.values())
    colors = plt.cm.Set3(range(len(labels)))  # Use a colormap for consistent colors

    # Create the bar chart
    bars = plt.bar(labels, sizes, color=colors)
    plt.title('Distribution of Part-of-Speech Classes in Failed Word Forms', fontsize=14, fontweight='bold')
    plt.xlabel('Part-of-Speech Tags', fontsize=12)
    plt.ylabel('Number of Words', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('plots/pos.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to only save, not display

    # Print summary statistics
    print("POS Tag Distribution:")
    print("-" * 30)
    total_words = sum(pos_counts.values())
    for pos, count in pos_counts.most_common():
        percentage = (count / total_words) * 100
        print(f"{pos:8s}: {count:4d} ({percentage:5.1f}%)")
    print("-" * 30)
    print(f"Total:   {total_words:4d} (100.0%)")

    print("\n" + "="*50)
    print("TOP 10 MOST COMMON FAILED WORDS:")
    print("="*50)
    for i, (word, count) in enumerate(word_counts.most_common(10), 1):
        print(f"{i:2d}. {word:20s} - {count:3d} occurrences")
    print("="*50)

    print("\n" + "="*50)
    print("TOP 10 MOST COMMON FAILED LEMMATA:")
    print("="*50)
    for i, (lemma, count) in enumerate(lemma_counts.most_common(10), 1):
        print(f"{i:2d}. {lemma:20s} - {count:3d} occurrences")
    print("="*50)

    print("\n" + "="*50)
    print("TOP 10 MOST COMMON FAILED SYLLABLES:")
    print("="*50)
    for i, (syllable, count) in enumerate(syllable_counts.most_common(10), 1):
        percentage = (count / len(failed_syllables)) * 100
        print(f"{i:2d}. {syllable:15s} - {count:3d} occurrences ({percentage:.1f}%)")
    print("="*50)