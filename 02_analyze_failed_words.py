import grc_odycy_joint_trf
import csv

# Load the odycy NLP pipeline
nlp = grc_odycy_joint_trf.load()

# Read the failed words TSV file
input_file = "failed_words/failed_words.tsv"
words = []
rows = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        # Strip ^ and _ characters from the word before processing
        word = row['failed_word'].replace('^', '').replace('_', '')
        words.append(word)
        rows.append(row)

# Process words with odycy pipeline
print(f"Processing {len(words)} words with odycy pipeline...")
processed_docs = list(nlp.pipe(words))

# Extract morphological analysis, lemma, and POS information
morph_analyses = []
lemmas = []
pos_tags = []

for doc in processed_docs:
    if len(doc) > 0:
        token = doc[0]  # Get the first (and likely only) token
        # Get morphological features
        morph = token.morph.to_dict() if token.morph else {}
        morph_str = "|".join([f"{k}={v}" for k, v in morph.items()]) if morph else ""
        morph_analyses.append(morph_str)
        
        # Get lemma
        lemma = token.lemma_ if token.lemma_ else ""
        lemmas.append(lemma)
        
        # Get POS tag
        pos = token.pos_ if token.pos_ else ""
        pos_tags.append(pos)
    else:
        morph_analyses.append("")
        lemmas.append("")
        pos_tags.append("")

# Add new columns to the existing data
for i, row in enumerate(rows):
    row['morphological_analysis'] = morph_analyses[i]
    row['lemma'] = lemmas[i]
    row['pos'] = pos_tags[i]

# Save the updated data
output_file = "failed_words/failed_words_with_analysis.tsv"
fieldnames = list(rows[0].keys()) if rows else []

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)

print(f"Results saved to {output_file}")
print(f"Added morphological analysis, lemma, and POS for {len(words)} words")