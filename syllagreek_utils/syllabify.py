# Define the set of Greek consonants used for syllable division.
CONSONANTS = set('βγδθκπτφχλρσμν')

def syllabify(tokens):
    """
    Divides a sequence of Greek tokens (letters or diphthongs) into syllables.

    Rules for syllabification:
    - A syllable must have a vowel or diphthong as its nucleus.
    - One consonant before a vowel becomes the onset of the syllable.
    - When multiple consonants are between vowels:
      - The first consonant joins the coda of the preceding syllable.
      - The remaining consonants form the onset of the next syllable.
    - Any consonants left at the end (no following vowel) are attached to the last syllable.

    Args:
        tokens (list of str): A list of single Greek letters or combined diphthongs.

    Returns:
        list of list of str: A list of syllables, where each syllable is itself a list of tokens.
    """
    syllables = []
    i = 0
    n = len(tokens)

    while i < n:
        current = []

        # Step 1: Collect any consonants before a vowel (possible onset).
        while i < n and tokens[i] in CONSONANTS:
            current.append(tokens[i])
            i += 1

        # Step 2: If we reach the end without encountering a vowel:
        if i >= n:
            if syllables:
                syllables[-1].extend(current)  # Attach to previous syllable
            else:
                syllables.append(current)      # Start a new syllable
            break

        # Step 3: Add the vowel (or diphthong) as the nucleus.
        current.append(tokens[i])
        i += 1

        # Step 4: Check upcoming consonants to decide syllable boundary.
        start = i
        count = 0
        while i < n and tokens[i] in CONSONANTS:
            count += 1
            i += 1

        if count == 0:
            # No consonants after nucleus → complete syllable
            syllables.append(current)
        elif count == 1:
            # One consonant after nucleus → assign to next syllable
            syllables.append(current)
            i = start  # Move back to the consonant to process next syllable
        else:
            # Two or more consonants after nucleus → split:
            # Attach first consonant to coda of current syllable,
            # remaining consonants start the next syllable.
            current.append(tokens[start])
            syllables.append(current)
            i = start + 1  # Continue from second consonant

    return syllables

def syllabify_joined(tokens):
    """
    Divides Greek tokens into syllables and joins the syllables into strings.

    Args:
        tokens (list of str): A list of single Greek letters or diphthongs.

    Returns:
        list of str: A list of syllable strings.
    """
    syllable_lists = syllabify(tokens)
    return [''.join(syllable) for syllable in syllable_lists]