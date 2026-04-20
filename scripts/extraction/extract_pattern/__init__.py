from .bit_prompt import EXTRACTOR as BIT_EXTRACTOR
from .equation_prompt import EXTRACTOR as EQUATION_EXTRACTOR
from .gravity_prompt import EXTRACTOR as GRAVITY_EXTRACTOR
from .roman_prompt import EXTRACTOR as ROMAN_EXTRACTOR
from .text_prompt import EXTRACTOR as TEXT_EXTRACTOR
from .unit_prompt import EXTRACTOR as UNIT_EXTRACTOR


def get_extractors():
    return [
        BIT_EXTRACTOR,
        TEXT_EXTRACTOR,
        ROMAN_EXTRACTOR,
        UNIT_EXTRACTOR,
        GRAVITY_EXTRACTOR,
        EQUATION_EXTRACTOR,
    ]
