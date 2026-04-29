from __future__ import annotations

import random

from scripts.gen_data.types.base import RawProblemRecord, TypeDataBuilder
from scripts.gen_data.types.bit_manipulation import BitManipulationBuilder
from scripts.gen_data.types.cipher import CipherBuilder
from scripts.gen_data.types.equation import CryptarithmBuilder, NumericEquationBuilder
from scripts.gen_data.types.gravity import GravityBuilder
from scripts.gen_data.types.numeral import NumeralBuilder
from scripts.gen_data.types.unit_conversion import UnitConversionBuilder


BUILDERS: tuple[TypeDataBuilder, ...] = (
    BitManipulationBuilder(),
    CipherBuilder(),
    NumeralBuilder(),
    UnitConversionBuilder(),
    GravityBuilder(),
    NumericEquationBuilder(),
    CryptarithmBuilder(),
)


def build_record(row: dict[str, str]) -> RawProblemRecord:
    prompt = row.get("prompt", "")
    for builder in BUILDERS:
        if builder.matches(prompt):
            return builder.build(row)
    raise ValueError(f"Unknown problem type for id={row.get('id', '')!r}")


def generate_record(
    category: str,
    rng: random.Random,
    problem_id: str,
) -> RawProblemRecord:
    for builder in BUILDERS:
        if category in builder.generated_categories:
            return builder.generate(category, rng, problem_id)
    raise ValueError(f"Unknown generated category: {category!r}")
