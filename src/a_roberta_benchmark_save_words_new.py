'''
Benchmarking the macronizer--mini RoBERTa syllable vowel-length classifier on the Norma Syllabarum Graecarum manually marked-up dataset.
We evaluate vowel length markup of alpha, iota and ypsilon (the dichrona) and not general scansion capabilities,
which means we try to avoid including failures due to the model not being trained on sandhi phenomena such as homosyllabic muta-cum-liquida, correption of long vowels etc. 

Hence we want stats only on how many open syllables with true dichrona get the correct syllable weight (heavy or light).
'''

import os
import re
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.nn.functional import softmax
from transformers import PreTrainedTokenizerFast, RobertaForTokenClassification

from syllagreek_utils import preprocess_greek_line, syllabify_joined
from grc_utils import is_vowel, word_with_real_dichrona, has_ambiguous_dichrona_in_open_syllables, lower_grc

to_clean = r'[\u0387\u037e\u00b7\.,!?;:\"()\[\]{}<>«»\-—…|⏑⏓†× ]'

def open_dichronic_syll(syll):
    '''
    Boolean check for the open dichronic sylls we want stats for.
    '''
    final_char = syll[-1]
    open_syll = is_vowel(final_char)
    dichronic = word_with_real_dichrona(syll)

    return open_syll and dichronic

def main(temp=1.0, clean=True):

    # -------- Load model --------
    model_path = "Ericu950/macronizer_mini"

    print(f"Benchmarking {model_path} on Norma Syllabarum Graecarum...")

    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(model_path)
    model: RobertaForTokenClassification = RobertaForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_len = min(
        getattr(tokenizer, "model_max_length", 512),
        getattr(model.config, "max_position_embeddings", 514) - 2
    )
    if max_len <= 0:
        max_len = 512

    # -------- Helper: parse gold labels in files --------
    def extract_expected_pattern(line):
        expected_labels = []
        tokens = []
        pattern = re.compile(r'(\[[^]]+\])|(\{[^}]+\})|([^\[\]\{\}]+)')  # 3 groups: [..], {..}, other
        for match in pattern.finditer(line):
            if match.group(1):
                content = match.group(1)[1:-1]
                tokens.append(content)
                expected_labels.append("H")
            elif match.group(2):
                content = match.group(2)[1:-1]
                tokens.append(content)
                expected_labels.append("L")
            elif match.group(3):
                tokens.append(match.group(3))
        raw_input = ''.join(tokens)
        return raw_input, expected_labels

    # -------- Rule-based syllable classification --------
    def classify_syllables(syllables, clear_mask):
        definitely_heavy_set = set("ὖὗἆἇἶἷήηωώἠἡἢἣἤἥἦἧὠὡὢὣὤὥὦὧὴὼᾄᾅᾆᾇᾐᾑᾔᾕᾖᾗᾠᾤᾦᾧᾳᾴᾶᾷῂῃῄῆῇῖῦῲῳῴῶῷ")
        ambiguous_set = set("ΐάίΰαιυϊϋύἀἁἂἃἄἅἰἱἲἳἴἵὐὑὓὔὕὰῒὶὺ")
        light_set = set("έεοόἐἑἓἔἕὀὁὂὃὄὅὲὸ")
        mute_consonants = set("βγδθκπτφχ")
        nonmute_consonants = set("λρμν")
        sigma = set("σ")
        all_consonants = mute_consonants | nonmute_consonants | sigma

        def token_contains(token, char_set):
            return any(ch in char_set for ch in token)

        def get_nucleus(syl):
            nucleus_chars = [ch for token in syl for ch in token if ch not in all_consonants]
            return ''.join(nucleus_chars) if nucleus_chars else None

        def classify_single_syllable(syl, next_syl):
            nucleus = get_nucleus(syl)
            if nucleus is None:
                return "light", False

            if len(nucleus) >= 2:
                base_class = "heavy"
            elif token_contains(nucleus, definitely_heavy_set):
                base_class = "heavy"
            elif token_contains(nucleus, ambiguous_set):
                base_class = "ambiguous"
            elif token_contains(nucleus, light_set):
                base_class = "light"
            else:
                base_class = "light"

            final_char = syl[-1][-1]
            
            # Track if final character is in ambiguous set
            has_ambiguous_final_char = final_char in ambiguous_set

            if base_class == "heavy":
                return "heavy", has_ambiguous_final_char
            elif base_class == "ambiguous":
                if final_char in nonmute_consonants:
                    return "heavy", has_ambiguous_final_char
                if final_char in mute_consonants and next_syl is not None:
                    next_onset = next_syl[0][0]
                    if next_onset not in nonmute_consonants:
                        return "heavy", has_ambiguous_final_char
                return "muta cum liquida", has_ambiguous_final_char
            elif base_class == "light":
                if final_char in nonmute_consonants or final_char in sigma:
                    return "heavy", has_ambiguous_final_char
                elif final_char in mute_consonants and next_syl is not None:
                    next_onset = next_syl[0][0]
                    if next_onset in nonmute_consonants:
                        return "muta cum liquida", has_ambiguous_final_char
                    else:
                        return "heavy", has_ambiguous_final_char
                else:
                    return "light", has_ambiguous_final_char

        classifications = []
        ambiguous_final_char_mask = []
        for i, syl in enumerate(syllables):
            if not clear_mask[i]:
                classifications.append(None)
                ambiguous_final_char_mask.append(False)
                continue
            next_syl = syllables[i+1] if i < len(syllables) - 1 else None
            result, has_ambiguous_final_char = classify_single_syllable(syl, next_syl)
            classifications.append(result)
            ambiguous_final_char_mask.append(has_ambiguous_final_char)
        return classifications, ambiguous_final_char_mask

    # -------- Predict syllable weights --------
    def predict_syllable_weights(raw_line):
        """
        Predict syllable weights for a line of Greek text.
        
        Returns:
            tuple: (final_labels, open_dichronic_mask)
                - final_labels: List of predicted labels ("H", "L", or None)
                - open_dichronic_mask: List of booleans indicating which syllables are 
                open dichronic syllables (what we want to evaluate)
        """
        # if clean: 
        #     raw_line = re.sub(to_clean, '', lower_grc(raw_line))
        tokens = preprocess_greek_line(raw_line)
        syllables = syllabify_joined(tokens)

        enc = tokenizer(
            syllables,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False
        )
        enc.pop("token_type_ids", None)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            temperature = temp
            probs = softmax(outputs.logits / temperature, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)[0].cpu().tolist()

        word_ids = tokenizer(
            syllables,
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_offsets_mapping=False
        ).word_ids()

        aligned_preds = []
        syllable_index = 0
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != syllable_index:
                syllable_index = wid
            if syllable_index < len(syllables):
                aligned_preds.append((syllables[syllable_index], pred_ids[i]))

        aligned_preds = aligned_preds[:len(syllables)]
        only_sylls = [s for s, _ in aligned_preds]
        labels = [l for _, l in aligned_preds]

        clear_mask = [l == 0 for l in labels]
        syllables_tokenized = [[ch for ch in syl] for syl in only_sylls]
        rule_based, _ = classify_syllables(syllables_tokenized, clear_mask)

        # Create mask for open dichronic syllables
        open_dichronic_mask = [open_dichronic_syll(syll) for syll in only_sylls]

        final_labels = []
        for model_label, rule in zip(labels, rule_based):
            if model_label == 1:
                final_labels.append("H")
            elif model_label == 2:
                final_labels.append("L")
            elif model_label == 0:
                if rule == "heavy":
                    final_labels.append("H")
                elif rule == "light":
                    final_labels.append("L")
                elif rule == "muta cum liquida":
                    final_labels.append(None)
                else:
                    final_labels.append(None)
            else:
                final_labels.append(None)
        return final_labels, open_dichronic_mask

    # ---------------------------------------------------------------------------
    # -------- Evaluation Loop (compute predictions on the fly) -----------------
    # ---------------------------------------------------------------------------

    gold_dir = "norma-syllabarum-graecarum/final"
    failed_words_dir = "failed_words"
    os.makedirs(failed_words_dir, exist_ok=True)

    results = defaultdict(lambda: {
        "H_correct":0, "H_total":0,
        "L_correct":0, "L_total":0,
        "all_correct":0, "all_total":0
    })

    failed_rows = []

    for filename in tqdm(os.listdir(gold_dir)):
        if not filename.endswith(".txt"):
            continue
        gold_path = os.path.join(gold_dir, filename)

        with open(gold_path, encoding="utf-8") as gf:
            for linenr, line in enumerate(gf, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Extract expected pattern BEFORE cleaning to preserve markup
                raw_input, expected = extract_expected_pattern(line)
                
                # NOTE Albin 10/09: Trying to avoid out-of-vocab tokens
                # Clean the raw_input (which already has markup removed) instead of the original line
                raw_input = lower_grc(raw_input)
                raw_input = re.sub(to_clean, '', raw_input)
                predicted_labels, open_dichronic_mask = predict_syllable_weights(raw_input)

                # Stats - only count open dichronic syllables
                # This focuses evaluation specifically on open syllables with true dichrona
                # to test vowel length markup accuracy while avoiding scansion complications
                for exp, pred_lbl, is_open_dichronic in zip(expected, predicted_labels, open_dichronic_mask):
                    if pred_lbl is None or not is_open_dichronic:
                        continue
                    
                    # Only count open dichronic syllables
                    results[filename]["all_total"] += 1
                    results[filename][f"{exp}_total"] += 1
                    if exp == pred_lbl:
                        results[filename]["all_correct"] += 1
                        results[filename][f"{exp}_correct"] += 1

                # Record failing sentences for failed words analysis
                fully_correct = all((p == e or p is None) for p, e in zip(predicted_labels, expected))
                if not fully_correct:
                    # ---- Build TSV rows for failed words ----
                    gold_pattern_str = ''.join(expected)
                    pred_pattern_str = ''.join([p if p else "-" for p in predicted_labels])

                    # Build syllable-to-word mapping
                    syll_to_word = []
                    
                    # Get clean text without markup to identify word boundaries
                    clean_text = re.sub(r'[\[\]\{\}]', '', line)
                    words = clean_text.split()
                    
                    # For each word, determine how many syllables it has and map them
                    for word in words:
                        word_sylls = syllabify_joined(preprocess_greek_line(word))
                        syll_to_word.extend([word] * len(word_sylls))

                    # Find failing syllables that are also open dichronic
                    failed_indices = []
                    for i, (g_lbl, p_lbl, is_open_dichronic) in enumerate(zip(expected, predicted_labels, open_dichronic_mask)):
                        if p_lbl != g_lbl and is_open_dichronic:
                            failed_indices.append(i)

                    # Map each failing syllable to its word with new rules
                    for idx in failed_indices:
                        if idx >= len(syll_to_word):
                            continue
                        
                        # Get the syllable that failed
                        raw_input_sylls = syllabify_joined(preprocess_greek_line(raw_input))
                        if idx >= len(raw_input_sylls):
                            continue
                        failed_syll = raw_input_sylls[idx]
                        
                        # Check if syllable contains a space (indicating word boundary within syllable)
                        if ' ' in failed_syll:
                            # Split at space and check the part before the space
                            before_space = failed_syll.split(' ')[0]
                            after_space = ' '.join(failed_syll.split(' ')[1:])
                            
                            # Check if the part before space has any vowels
                            has_vowel_before = any(is_vowel(char) for char in before_space)
                            
                            if has_vowel_before:
                                # Use the word that the part before space belongs to
                                failed_word = syll_to_word[idx]
                            else:
                                # Use the word that the part after space belongs to
                                # Find which word the after_space part belongs to
                                failed_word = None
                                for word in words:
                                    if after_space in word:
                                        failed_word = word
                                        break
                                if failed_word is None:
                                    failed_word = syll_to_word[idx]  # fallback
                        else:
                            # Regular syllable without space
                            failed_word = syll_to_word[idx]
                        
                        # Check if the failed word contains any vowels and has ambiguous dichrona in open syllables
                        # Skip words without vowels (like "δ'") and words that don't meet our criteria
                        if any(is_vowel(char) for char in failed_word) and has_ambiguous_dichrona_in_open_syllables(failed_word):
                            failed_rows.append((failed_syll, failed_word, raw_input, gold_pattern_str, pred_pattern_str))

    # Write TSV for failed words
    outpath_words = os.path.join(failed_words_dir, "failed_words.tsv")
    with open(outpath_words, "w", encoding="utf-8") as out:
        out.write("failed_syllable\tfailed_word\tgold_line\tgold_pattern\tpredicted_pattern\n")
        for row in failed_rows:
            out.write("\t".join(row) + "\n")

    # -------- Reporting --------
    overall = defaultdict(int)

    print(f"\n{'File':<30} {'All Acc':>8} {'H Acc':>8} {'L Acc':>8}")
    print("-"*60)
    for file, data in sorted(results.items()):
        all_acc = data["all_correct"]/data["all_total"] if data["all_total"] else 0
        h_acc = data["H_correct"]/data["H_total"] if data["H_total"] else 0
        l_acc = data["L_correct"]/data["L_total"] if data["L_total"] else 0
        print(f"{file:<30} {all_acc:8.2%} {h_acc:8.2%} {l_acc:8.2%}")
        for k in ["all_correct","all_total","H_correct","H_total","L_correct","L_total"]:
            overall[k] += data[k]

    print(f"\nEvaluation limited to open dichronic syllables (open syllables with α, ι, υ):")
    print(f"Total syllables evaluated: {overall['all_total']} (H: {overall['H_total']}, L: {overall['L_total']})")

    print("\nOverall Accuracy:")
    all_acc = overall["all_correct"]/overall["all_total"] if overall["all_total"] else 0
    h_acc = overall["H_correct"]/overall["H_total"] if overall["H_total"] else 0
    l_acc = overall["L_correct"]/overall["L_total"] if overall["L_total"] else 0
    print(f"{'All':<10}: {all_acc:.2%}")
    print(f"{'Heavy':<10}: {h_acc:.2%}")
    print(f"{'Light':<10}: {l_acc:.2%}")

    # -------- Validation Test --------
    print("\n" + "="*60)
    print("VALIDATION TEST: Checking failed_words.tsv with has_ambiguous_dichrona_in_open_syllables")
    print("="*60)

    # Read and test each word in the failed_words.tsv
    validation_passed = 0
    validation_failed = 0
    failed_examples = []

    with open(outpath_words, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # Skip header
        for i, line in enumerate(lines):
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[1]  # failed_word is now in the second column
                    if has_ambiguous_dichrona_in_open_syllables(word.replace("^", "").replace("_", "")):
                        validation_passed += 1
                    else:
                        validation_failed += 1
                        if len(failed_examples) < 10:  # Keep first 10 examples
                            failed_examples.append(word)

    print(f"Words that PASS has_ambiguous_dichrona_in_open_syllables: {validation_passed}")
    print(f"Words that FAIL has_ambiguous_dichrona_in_open_syllables: {validation_failed}")

    if validation_failed > 0:
        print(f"\nFIRST {min(len(failed_examples), 10)} FAILING EXAMPLES:")
        for word in failed_examples:
            print(f"  {word}")
        print("\n❌ VALIDATION FAILED: Some words in failed_words.tsv don't pass has_ambiguous_dichrona_in_open_syllables")
    else:
        print("\n✅ VALIDATION PASSED: All words in failed_words.tsv pass has_ambiguous_dichrona_in_open_syllables")

if __name__ == "__main__":
    main(temp=1.0)