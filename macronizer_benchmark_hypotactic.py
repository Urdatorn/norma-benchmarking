# -*- coding: utf-8 -*-
"""
Testing macronized and unmacronized rule-based scansion against Hypotactic
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import signal
import sys
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from grc_utils import DICHRONA, is_vowel, lower_grc, short_set, syllabifier, is_diphthong, short_set
from syllagreek_utils import preprocess_greek_line, syllabify_joined

print("Imports complete.")

def handle_sigint(sig, frame):
    print("\nInterrupted with Ctrl+C. Exiting cleanly...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

def load_macronizer(model_path="./macronizer_mini"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForTokenClassification.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device

model, tokenizer, device = load_macronizer()

def macronizer_scan(line):

  tokens = preprocess_greek_line(line)
  syllables = syllabify_joined(tokens)

  tokenized = tokenizer(
      syllables,
      is_split_into_words=True,
      return_tensors="pt",
      truncation=True,
      max_length=512,
      padding="max_length"
  )

  # RoBERTa doesn't use token_type_ids, but remove if present to be safe
  if "token_type_ids" in tokenized:
      del tokenized["token_type_ids"]

  inputs = {k: v.to(device) for k, v in tokenized.items()}

  # -------- Run inference to get pred_ids --------
  with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits  # [batch, seq_len, num_labels]
      probs = softmax(logits, dim=-1)
      pred_ids = torch.argmax(probs, dim=-1).squeeze(0).cpu().tolist()

  # -------- Align Predictions with Syllables (first subtoken per word) --------
  # Preferred: use BatchEncoding.word_ids(batch_index=0)
  word_ids = tokenized.word_ids(batch_index=0)

  aligned_preds = []
  seen = set()
  for i, w_id in enumerate(word_ids):
      if w_id is None or w_id in seen:
          continue
      aligned_preds.append((syllables[w_id], pred_ids[i]))
      seen.add(w_id)
      if len(aligned_preds) == len(syllables):
          break

  # -------- Rule-based Postprocessing for "clear" syllables --------
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
              return "u"

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

          if base_class == "heavy":
              return "-"
          elif base_class == "ambiguous":
              if final_char in nonmute_consonants:
                  return "-"
              if final_char in mute_consonants and next_syl is not None:
                  next_onset = next_syl[0][0]
                  if next_onset not in nonmute_consonants:
                      return "-"
              return "-" # forcing heteroclitic mcl
          elif base_class == "light":
              if final_char in nonmute_consonants or final_char in sigma:
                  return "-"
              elif final_char in mute_consonants and next_syl is not None:
                  next_onset = next_syl[0][0]
                  if next_onset in nonmute_consonants:
                      return "-" # forcing heteroclitic mcl
                  else:
                      return "-"
              else:
                  return "u"
          else:
            print("error: no base class")
            return "u" # in case of no base_class, default to short


      classifications = []
      for i, syl in enumerate(syllables):
          if not clear_mask[i]:
              classifications.append(None)
              continue
          next_syl = syllables[i+1] if i < len(syllables) - 1 else None
          classifications.append(classify_single_syllable(syl, next_syl))

      return classifications

  # -------- Prepare Data for Classification --------
  only_sylls = [s for s, _ in aligned_preds]
  labels = [l for _, l in aligned_preds]
  clear_mask = [l == 0 for l in labels]  # assumes 0="clear"
  syllables_tokenized = [[ch for ch in syl] for syl in only_sylls]

  # -------- Apply Rule-based Classifier --------
  rule_based = classify_syllables(syllables_tokenized, clear_mask)

  # -------- Output --------

  scansion = []
  for syl, label, rule_label in zip(only_sylls, labels, rule_based):
      if label == 1:
          #print(f"{syl:>10} → ambiguous → -")
          scansion.append("-")
      elif label == 2:
          #print(f"{syl:>10} → ambiguous → u")
          scansion.append("u")
      elif label == 0:
          result = rule_label or "u" # ^ if no label
          #print(f"{syl:>10} → clear → {result}")
          scansion.append(result)
      else:
          scansion.append("u") # defaulting to short if model does not predict
  return "".join(scansion)

#line = "Εὖθ᾽ ὑπὸ Πηλείωνιδά μηθεοείκελος Ἕκτωρ"
#predicted_scan = macronizer_scan(line)
#print(predicted_scan)

to_clean = r'[\u0387\u037e\u00b7\.,!?;:\"()\[\]{}<>«»\-—…|⏑⏓†× ]'

def clean_and_lower(input):
  lowered = lower_grc(input)
  cleaned = re.sub(to_clean, '', lowered)

  return cleaned

"""We let the baseline be a simple rule-based scansion that treats all dichrona as short."""

def short_vowel(syllable):
    short_set_with_dichrona = short_set | DICHRONA
    return any(vowel in syllable for vowel in short_set_with_dichrona)

def heavy_syll(syll):
    """Check if a syllable is heavy (either ends on a consonant or contains a long vowel/diphthong)."""

    closed = not is_vowel(syll[-1])

    substrings = [syll[i:i+2] for i in range(len(syll) - 1)]
    has_diphthong = any(is_diphthong(substring) for substring in substrings)

    has_long = not short_vowel(syll) # short_vowel does not include short dichrona

    return closed or has_diphthong or has_long

def scan_rules(line):
    line = clean_and_lower(line)
    sylls = syllabifier(line) # a list of syllable strings, e.g. ['οἳδ᾽', 'ἐ', 'πὶ']
    #print(sylls)

    scansion = []
    for syll in sylls:
      if heavy_syll(syll):
        scansion.append("-")
      else:
        scansion.append("u")

    return "".join(scansion)

#line = "Εὖθ᾽ ὑπὸ Πηλείωνιδά μηθεοείκελος Ἕκτωρ"
#print(scan_rules(line)) # -uu---uu-uu-uu--

#!git clone https://github.com/Urdatorn/hypotactic.git

contents = re.compile(r'\[(.+?)\]')

def compare_predicted_scansion_with_gold(txt_files, debug=False):
  total_lines = 0
  failed_lines = 0
  failed_lines_baseline = 0

  for txt in tqdm(txt_files, total=len(txt_files), desc="Going through NSG..."):
    with open(txt, "r", encoding="utf-8") as file:
      lines = file.readlines()
      for line in lines:
        total_lines += 1

        tabs = re.findall(contents, line)
        greek_raw = tabs[0]
        greek = clean_and_lower(greek_raw)

        gold_scansion = tabs[1]
        gold_scansion = gold_scansion.replace(" ", "")
        predicted_scansion = macronizer_scan(greek)
        baseline_rule_scansion = scan_rules(greek)

        # removing last free syll (brevis in longo)
        gold_scansion = gold_scansion[:-1]
        predicted_scansion = predicted_scansion[:-1]
        baseline_rule_scansion = baseline_rule_scansion[:-1]

        if debug:
          print(greek_raw)
          print(f"\t {gold_scansion}")
          print(f"\t {predicted_scansion}")
          print(f"\t {baseline_rule_scansion}")

        if predicted_scansion != gold_scansion:
          failed_lines += 1

        if baseline_rule_scansion != gold_scansion:
          failed_lines_baseline += 1

        # option: char-by-char check (only works in isometric meters, i.e. not hexameter)
        #for gold, pred in zip(list(gold_scansion), list(predicted_scansion)):
        #  etc

  return failed_lines, failed_lines_baseline, total_lines


if __name__ == "__main__":
    try:
        # Kör jämförelsen på hela eller valfri del av hypotactic
        dir_path = "hypotactic/hypotactic_txts_greek/"
        txt_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]
        #txt_files = [os.path.join(dir_path, "cleanthes.txt")]

        failed_lines, failed_lines_baseline, total_lines = compare_predicted_scansion_with_gold(txt_files, debug=False)
        print(f"Macronized: Failed {failed_lines} out of {total_lines}, {failed_lines / total_lines}")
        print(f"Unmacronized baseline: Failed {failed_lines_baseline} out of {total_lines}, {failed_lines_baseline / total_lines}")
        with open("macronizer_benchmark_hypotactic.txt", "w", encoding="utf-8") as out_file:
            out_file.write(f"Macronized: Failed {failed_lines} out of {total_lines}, {failed_lines / total_lines}\n")
            out_file.write(f"Unmacronized baseline: Failed {failed_lines_baseline} out of {total_lines}, {failed_lines_baseline / total_lines}\n")
    except KeyboardInterrupt:
        # Fallback if signal handler didn’t catch it
        print("\nInterrupted. Exiting...")
        sys.exit(0)