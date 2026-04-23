from scripts.cot_prompt.bit_manipulation import reasoning_bit_manipulation
from scripts.cot_prompt.cipher import reasoning_cipher
from scripts.cot_prompt.cryptarithm import reasoning_cryptarithm
from scripts.cot_prompt.equation_numeric import reasoning_equation_numeric
from scripts.cot_prompt.gravity import reasoning_gravity
from scripts.cot_prompt.numeral import reasoning_numeral
from scripts.cot_prompt.unit_conversion import reasoning_unit_conversion

GENERATORS = {
    "gravity": reasoning_gravity,
    "unit_conversion": reasoning_unit_conversion,
    "cipher": reasoning_cipher,
    "bit_manipulation": reasoning_bit_manipulation,
    "numeral": reasoning_numeral,
    "equation_numeric_deduce": reasoning_equation_numeric,
    "equation_numeric_guess": reasoning_equation_numeric,
    "cryptarithm_deduce": reasoning_cryptarithm,
    "cryptarithm_guess": reasoning_cryptarithm,
}
INVESTIGATION_CATEGORIES: set[str] = {
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
}
SKIP_CATEGORIES: set[str] = set()