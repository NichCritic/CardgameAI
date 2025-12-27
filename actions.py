from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ActionType(Enum):
    DRAW_CARD = "draw_card"
    PLAY_POKEMON = "play_pokemon"
    ATTACH_ENERGY = "attach_energy"
    ATTACK = "attack"
    END_TURN = "end_turn"
    PASS = "pass"


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    hand_index: Optional[int] = None
    pokemon_index: Optional[int] = None
    bench: bool = False
