from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CardType(Enum):
    POKEMON = "pokemon"
    ENERGY = "energy"
    TRAINER = "trainer"


class EnergyType(Enum):
    FIRE = "fire"
    WATER = "water"
    GRASS = "grass"
    ELECTRIC = "electric"
    PSYCHIC = "psychic"
    FIGHTING = "fighting"
    DARK = "dark"
    STEEL = "steel"
    COLORLESS = "colorless"


@dataclass(frozen=True)
class PokemonCard:
    name: str
    hp: int
    energy_types: tuple[EnergyType, ...]
    attack_cost: tuple[EnergyType, ...]
    attack_damage: int
    retreat_cost: int


@dataclass(frozen=True)
class EnergyCard:
    energy_type: EnergyType


@dataclass(frozen=True)
class TrainerCard:
    name: str
    effect: str


Card = PokemonCard | EnergyCard | TrainerCard
