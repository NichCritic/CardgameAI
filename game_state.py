from dataclasses import dataclass, field
from typing import Optional
from cards import Card, PokemonCard, EnergyCard


@dataclass
class PokemonInPlay:
    card: PokemonCard
    attached_energy: list[EnergyCard] = field(default_factory=list)
    damage: int = 0
    status: Optional[str] = None

    @property
    def is_knocked_out(self) -> bool:
        return self.damage >= self.card.hp


@dataclass
class PlayerState:
    deck: list[Card]
    hand: list[Card] = field(default_factory=list)
    active_pokemon: Optional[PokemonInPlay] = None
    bench: list[PokemonInPlay] = field(default_factory=list)
    prizes: list[Card] = field(default_factory=list)
    discard: list[Card] = field(default_factory=list)
    energy_attached_this_turn: int = 0
    pokemon_played_this_turn: bool = False

    def reset_turn_flags(self) -> None:
        self.energy_attached_this_turn = 0
        self.pokemon_played_this_turn = False


@dataclass
class GameState:
    player1: PlayerState
    player2: PlayerState
    current_player: int
    turn_number: int
    winner: Optional[int] = None

    @property
    def current_player_state(self) -> PlayerState:
        return self.player1 if self.current_player == 0 else self.player2

    @property
    def opponent_player_state(self) -> PlayerState:
        return self.player2 if self.current_player == 0 else self.player1
