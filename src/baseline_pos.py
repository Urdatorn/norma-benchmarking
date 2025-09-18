import os
import re
import grc_odycy_joint_trf
from collections import Counter

from grc_utils import count_ambiguous_dichrona_in_open_syllables

def get_baseline_pos_distribution():
    """
    Returns a dictionary with POS tag counts from the baseline corpus.
    Only includes tokens that have at least one ambiguous dichrona in open syllables.
    This can be imported and used as a reference in other scripts.
    """
    nlp = grc_odycy_joint_trf.load()

    dir = "norma-syllabarum-graecarum/final/"
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".txt")]

    sentences = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        to_clean = r'[\u0387\u037e\u00b7\.,!?;:\"()\[\]{}<>«»\-—…|⏑⏓†×^_]'
        text = re.sub(to_clean, '', text)
        sentences.extend(text.split("."))

    processed_docs = list(nlp.pipe(sentences))

    # Extract POS tags from all processed documents
    pos_tags = []
    for doc in processed_docs:
        for token in doc:
            if token.pos_:  # Only include tokens with POS tags
                # Only include tokens that have ambiguous dichrona in open syllables
                if count_ambiguous_dichrona_in_open_syllables(token.text) >= 1:
                    pos_tags.append(token.pos_)
    
    # Count the frequency of each POS tag
    pos_counter = Counter(pos_tags)
    
    return dict(pos_counter)

def main():
    """
    Main function that prints statistics about the POS distribution.
    Only includes tokens with ambiguous dichrona in open syllables.
    """
    pos_counter = get_baseline_pos_distribution()
    
    # Print some statistics
    total_pos_tags = sum(pos_counter.values())
    print(f"Total tokens with POS tags (with ambiguous dichrona in open syllables): {total_pos_tags}")
    print(f"Number of unique POS tags: {len(pos_counter)}")
    print(f"Most common POS tags:")
    for pos, count in Counter(pos_counter).most_common(10):
        print(f"  {pos}: {count} ({count/total_pos_tags*100:.1f}%)")
    
    return pos_counter


if __name__ == "__main__":
    main()