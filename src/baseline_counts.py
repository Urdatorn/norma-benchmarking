import os
import re
from collections import Counter

def word_list(text):
    to_clean = r'[\[\]{}]'
    to_space = r'[\u0387\u037e\u00b7\.,!?;:\"()<>«»\-—…|⏑⏓†×]'  # NOTE hyphens must be escaped

    cleaned_text = re.sub(to_clean, '', text)
    cleaned_text = re.sub(to_space, ' ', cleaned_text)

    word_list = [word for word in cleaned_text.split() if word]

    return word_list

def extract_syllables(text):
    """Extract syllables from markup like [Ἦ] {πόλ} [λ' ἄ^] etc."""
    # Pattern to match content inside [] or {}
    pattern = r'[\[\{]([^\]\}]+)[\]\}]'
    syllables = re.findall(pattern, text)
    
    return syllables

def word_and_syll_baselines():
    input_dir = "norma-syllabarum-graecarum/final"
    
    all_words = []
    all_syllables = []
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                
                # Extract words
                words = word_list(text)
                print(words)
                all_words.extend(words)
                
                # Extract syllables
                syllables = extract_syllables(text)
                all_syllables.extend(syllables)
    
    # Count frequencies
    word_counts = Counter(all_words)
    syllable_counts = Counter(all_syllables)
    
    print(f"\nBASELINE STATISTICS:")
    print("=" * 50)
    print(f"Total unique words: {len(word_counts)}")
    print(f"Total word instances: {len(all_words)}")
    print(f"Total unique syllables: {len(syllable_counts)}")
    print(f"Total syllable instances: {len(all_syllables)}")
    
    print(f"\nTop 20 most common words:")
    print("-" * 40)
    for i, (word, count) in enumerate(word_counts.most_common(20), 1):
        percentage = (count / len(all_words)) * 100
        print(f"{i:2d}. {word:15s} - {count:5d} times ({percentage:.2f}%)")
    
    print(f"\nTop 20 most common syllables:")
    print("-" * 40)
    for i, (syllable, count) in enumerate(syllable_counts.most_common(20), 1):
        percentage = (count / len(all_syllables)) * 100
        print(f"{i:2d}. {syllable:15s} - {count:5d} times ({percentage:.2f}%)")
    
    print(f"\nBaseline counts saved to:")
    print(f"- baseline_word_counts.txt")
    print(f"- baseline_syllable_counts.txt")

    return word_counts, syllable_counts

if __name__ == "__main__":
    word_and_syll_baselines()
