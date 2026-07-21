"""16 deterministic morphological features per token.

Ported from the original system (src/embedding.rs, CharClassFeatures). The token id is
interpreted as a Unicode codepoint; for ids < 128 these are ASCII characters, and beyond
that we derive features from the numeric value.

These features have NO trainable parameters — they are computed deterministically, then
concatenated with the Fourier basis in FractalEmbedding.
"""

import torch


class CharClassFeatures:
    """Extraction of 16 morphological features from a token id.

    Features (index: meaning):
        0  : is_vowel          (a, e, i, o, u)
        1  : is_consonant      (non-vowel letter)
        2  : is_digit          (0-9)
        3  : is_space          (0x20)
        4  : is_uppercase
        5  : is_lowercase
        6  : is_punctuation    (!"#$%...)
        7  : is_alphabetic
        8  : is_numeric        (alias of is_digit here)
        9  : is_whitespace     (space, tab, newline)
        10 : is_control        (codepoint < 32 or == 127)
        11 : digit_value       (0-9, or 0 if not a digit)
        12 : char_category     (simplified Unicode category as float)
        13 : position_in_alphabet (0-25, or -1 if not a letter; we encode -1→0)
        14 : is_ascii          (codepoint < 128)
        15 : parity            (even token id = 1, odd = 0)
    """

    N_FEATURES = 16

    VOWELS = frozenset(b"aeiouAEIOU")

    @staticmethod
    def extract(token_id: int) -> torch.Tensor:
        """Returns a float32 tensor of shape (16,)."""
        f = torch.zeros(CharClassFeatures.N_FEATURES, dtype=torch.float32)

        # We interpret the low byte as a potential character.
        as_byte = (token_id & 0xFF)

        # 0: is_vowel
        is_vowel_bool = as_byte in CharClassFeatures.VOWELS
        f[0] = float(is_vowel_bool)

        # 1: is_consonant (alphabetic non-vowel letter)
        is_alpha = (
            (0x41 <= as_byte <= 0x5A) or  # A-Z
            (0x61 <= as_byte <= 0x7A)     # a-z
        )
        f[1] = float(is_alpha and not is_vowel_bool)

        # 2: is_digit
        is_digit = 0x30 <= as_byte <= 0x39
        f[2] = float(is_digit)

        # 3: is_space (0x20)
        f[3] = float(as_byte == 0x20)

        # 4: is_uppercase
        f[4] = float(0x41 <= as_byte <= 0x5A)

        # 5: is_lowercase
        f[5] = float(0x61 <= as_byte <= 0x7A)

        # 6: is_punctuation (ASCII punctuation ranges)
        is_punct = (
            (0x21 <= as_byte <= 0x2F) or
            (0x3A <= as_byte <= 0x40) or
            (0x5B <= as_byte <= 0x60) or
            (0x7B <= as_byte <= 0x7E)
        )
        f[6] = float(is_punct)

        # 7: is_alphabetic
        f[7] = float(is_alpha)

        # 8: is_numeric (alias of is_digit here)
        f[8] = float(is_digit)

        # 9: is_whitespace (space, tab 0x09, newline 0x0A, CR 0x0D)
        f[9] = float(as_byte in (0x09, 0x0A, 0x0D, 0x20))

        # 10: is_control (codepoint < 32 or == 127)
        f[10] = float(as_byte < 0x20 or as_byte == 0x7F)

        # 11: digit_value
        f[11] = float(as_byte - 0x30) if is_digit else 0.0

        # 12: char_category simplified: 1.0 letter, 2.0 digit, 3.0 punctuation,
        #     4.0 space, 5.0 control, 0.0 other.
        if is_alpha:
            f[12] = 1.0
        elif is_digit:
            f[12] = 2.0
        elif is_punct:
            f[12] = 3.0
        elif f[9] > 0.5:
            f[12] = 4.0
        elif f[10] > 0.5:
            f[12] = 5.0

        # 13: position_in_alphabet (0-25, or 0 if not a letter)
        if 0x41 <= as_byte <= 0x5A:
            f[13] = float(as_byte - 0x41)
        elif 0x61 <= as_byte <= 0x7A:
            f[13] = float(as_byte - 0x61)

        # 14: is_ascii
        f[14] = float(token_id < 128)

        # 15: parity (even token id)
        f[15] = float((token_id % 2) == 0)

        return f
