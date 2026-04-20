from .bit_manipulation import EXTRACTOR as BIT_EXTRACTOR
from .bit_manipulation import VALIDATOR as BIT_VALIDATOR
from .gravity_distance import EXTRACTOR as GRAVITY_EXTRACTOR
from .gravity_distance import VALIDATOR as GRAVITY_VALIDATOR
from .numeric_equation import EXTRACTOR as NUMERIC_EQUATION_EXTRACTOR
from .numeric_equation import VALIDATOR as NUMERIC_EQUATION_VALIDATOR
from .roman_numeral import EXTRACTOR as ROMAN_EXTRACTOR
from .roman_numeral import VALIDATOR as ROMAN_VALIDATOR
from .symbol_equation import EXTRACTOR as SYMBOL_EQUATION_EXTRACTOR
from .symbol_equation import VALIDATOR as SYMBOL_EQUATION_VALIDATOR
from .text_decryption import EXTRACTOR as TEXT_EXTRACTOR
from .text_decryption import VALIDATOR as TEXT_VALIDATOR
from .unit_conversion import EXTRACTOR as UNIT_EXTRACTOR
from .unit_conversion import VALIDATOR as UNIT_VALIDATOR


def get_extractors():
    return [
        BIT_EXTRACTOR,
        TEXT_EXTRACTOR,
        ROMAN_EXTRACTOR,
        UNIT_EXTRACTOR,
        GRAVITY_EXTRACTOR,
        NUMERIC_EQUATION_EXTRACTOR,
        SYMBOL_EQUATION_EXTRACTOR,
    ]


def get_validators():
    return [
        BIT_VALIDATOR,
        TEXT_VALIDATOR,
        ROMAN_VALIDATOR,
        UNIT_VALIDATOR,
        GRAVITY_VALIDATOR,
        NUMERIC_EQUATION_VALIDATOR,
        SYMBOL_EQUATION_VALIDATOR,
    ]
