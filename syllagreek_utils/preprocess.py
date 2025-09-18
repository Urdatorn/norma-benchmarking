# preprocess.py

import re

# === 1. Oxia → Tonos replacements ===
# These replace legacy Greek accents with the modern Unicode tonos versions
OXIA_TO_TONOS = {
    "ά": "ά",  # U+1F71 → U+03AC (alpha)
    "έ": "έ",  # U+1F73 → U+03AD (epsilon)
    "ή": "ή",  # U+1F75 → U+03AE (eta)
    "ί": "ί",  # U+1F77 → U+03AF (iota)
    "ύ": "ύ",  # U+1F7B → U+03CD (upsilon)
    "ό": "ό",  # U+1F79 → U+03CC (omicron)
    "ώ": "ώ",  # U+1F7D → U+03CE (omega)
}

# === 2. Diphthong component sets ===
diphth_y = {'α', 'ε', 'η', 'ο'}
upsilon_forms = {'ὐ','ὔ','υ','ὑ','ύ','ὖ','ῦ','ὕ','ὗ','ὺ','ὒ','ὓ'}

diphth_i = {'α', 'ε', 'ο', 'υ'}
iota_forms = {'ἰ','ί','ι','ῖ','ἴ','ἶ','ἵ','ἱ','ἷ','ὶ','ἲ','ἳ'}

# Iota subscript/adscript combinations
adscr_i_first = {
    'α','η','ω','ἀ','ἠ','ὠ','ἁ','ἡ','ὡ','ά','ή','ώ','ὰ','ὴ','ὼ',
    'ᾶ','ῆ','ῶ','ὤ','ὥ','ὢ','ὣ','ἄ','ἅ','ἂ','ἃ','ἤ','ἥ','ἣ',
    'ἢ','ἦ','ἧ','ἆ','ἇ','ὧ','ὦ'
}
adscr_i_second = {'ι'}

# === 3. Word processor: expansion and diphthong merging ===

def process_word(word):
    """
    Expand special Greek letters and merge diphthongs.

    Args:
        word (str): A lowercase Greek word.

    Returns:
        list of str: A list of tokens (letters or diphthongs).
    """
    expanded = []

    # Step 1: Expand characters like ζ → δσ, ξ → κσ, etc.
    for char in word:
        if char == 'ζ':
            expanded.extend(['δ', 'σ'])
        elif char == 'ς':
            expanded.append('σ')
        elif char == 'ῥ':
            expanded.append('ρ')
        elif char == 'ξ':
            expanded.extend(['κ', 'σ'])
        elif char == 'ψ':
            expanded.extend(['π', 'σ'])
        else:
            expanded.append(char)

    # Step 2: Merge diphthongs and adscript combinations
    combined = []
    i = 0
    while i < len(expanded):
        a = expanded[i]
        b = expanded[i+1] if i + 1 < len(expanded) else ''

        if a in diphth_y and b in upsilon_forms:
            combined.append(a + b)
            i += 2
        elif a in diphth_i and b in iota_forms:
            combined.append(a + b)
            i += 2
        elif a in adscr_i_first and b in adscr_i_second:
            combined.append(a + b)
            i += 2
        else:
            combined.append(a)
            i += 1

    return combined

# === 4. Accent Normalization ===

def replace_oxia_with_tonos(text):
    """
    Replace oxia accents in text with tonos equivalents using Unicode mapping.

    Args:
        text (str): Input Greek string.

    Returns:
        str: Normalized string with tonos accents.
    """
    return ''.join(OXIA_TO_TONOS.get(ch, ch) for ch in text)

# === 5. Full Preprocessor ===

def preprocess_greek_line(line):
    """
    Normalize, extract, and tokenize a line of Greek text.

    Steps:
    1. Normalize oxia to tonos.
    2. Extract valid Greek words and discard punctuation.
    3. Expand compound characters and merge diphthongs.
    4. Flatten the tokens across all words.

    Args:
        line (str): A full Greek sentence or phrase.

    Returns:
        list of str: A flat list of tokens (letters or diphthongs).
    """
    # Step 1: Replace oxia with tonos
    line = replace_oxia_with_tonos(line)

    # Step 2: Extract only Greek characters (ignore punctuation, numbers, etc.)
    words = re.findall(
        r"[ΆΐΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίΰαβγδεζηθικλμνξοπρςστυφχψωϊϋόύώ"
        r"ἀἁἂἃἄἅἆἇἈἉἊἋἌἍἎ"
        r"ἐἑἒἓἔἕἘἙἜἝ"
        r"ἠἡἢἣἤἥἦἧἨἩἪἫἬἭἮ"
        r"ἰἱἲἳἴἵἶἷἸἹἺἻἼἽἾ"
        r"ὀὁὂὃὄὅὈὉὊὋὌὍ"
        r"ὐὑὒὓὔὕὖὗὙὛὝ"
        r"ὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯ"
        r"ὰὲὴὶὸὺὼᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍ"
        r"ᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝ"
        r"ᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯ"
        r"ᾲᾳᾴᾶᾷῂῃῄῆῇῒῖῗῢῤῥῦῧῬῲῳῴῶῷ]+",
        line.lower()
    )

    # Step 3: Tokenize each word using expansion rules
    token_lists = [process_word(word) for word in words]

    # Step 4: Flatten token lists across all words
    tokens = [token for tokens in token_lists for token in tokens]

    return tokens