import random
from typing import Optional
from cards import Card, PokemonCard, EnergyCard, TrainerCard, EnergyType
from game_state import GameState, PlayerState, PokemonInPlay


def create_deck(pokemon_count: int = 20, energy_count: int = 20, trainer_count: int = 20) -> list[Card]:
    from cards import PokemonCard, EnergyCard, TrainerCard, EnergyType
    
    deck = []
    
    pokemon_cards = [
        PokemonCard("Pikachu", 60, (EnergyType.ELECTRIC,), (EnergyType.ELECTRIC, EnergyType.COLORLESS), 30, 1),
        PokemonCard("Charmander", 50, (EnergyType.FIRE,), (EnergyType.FIRE,), 20, 1),
        PokemonCard("Squirtle", 50, (EnergyType.WATER,), (EnergyType.WATER,), 20, 1),
        PokemonCard("Bulbasaur", 50, (EnergyType.GRASS,), (EnergyType.GRASS,), 20, 1),
        PokemonCard("Raichu", 80, (EnergyType.ELECTRIC,), (EnergyType.ELECTRIC, EnergyType.ELECTRIC), 50, 1),
    ]
    
    energy_types = list(EnergyType)
    trainer_cards = [
        TrainerCard("Potion", "Heal 20 damage"),
        TrainerCard("Switch", "Switch active Pokemon"),
        TrainerCard("Professor", "Draw 3 cards"),
    ]
    
    for _ in range(pokemon_count):
        deck.append(random.choice(pokemon_cards))
    for _ in range(energy_count):
        deck.append(EnergyCard(random.choice(energy_types)))
    for _ in range(trainer_count):
        deck.append(random.choice(trainer_cards))
    
    random.shuffle(deck)
    return deck


def initialize_game(deck1: Optional[list[Card]] = None, deck2: Optional[list[Card]] = None) -> GameState:
    if deck1 is None:
        deck1 = create_deck()
    if deck2 is None:
        deck2 = create_deck()
    
    assert len(deck1) >= 7, "Deck must have at least 7 cards"
    assert len(deck2) >= 7, "Deck must have at least 7 cards"
    
    player1_deck = deck1.copy()
    player2_deck = deck2.copy()
    
    player1_hand = [player1_deck.pop() for _ in range(7)]
    player2_hand = [player2_deck.pop() for _ in range(7)]
    
    prize_count = 6
    player1_prizes = [player1_deck.pop() for _ in range(prize_count)]
    player2_prizes = [player2_deck.pop() for _ in range(prize_count)]
    
    return GameState(
        player1=PlayerState(deck=player1_deck, hand=player1_hand, prizes=player1_prizes),
        player2=PlayerState(deck=player2_deck, hand=player2_hand, prizes=player2_prizes),
        current_player=0,
        turn_number=1
    )


def draw_card(state: GameState, player_idx: int) -> bool:
    player = state.player1 if player_idx == 0 else state.player2
    assert len(player.deck) > 0, "Cannot draw from empty deck"
    
    card = player.deck.pop()
    player.hand.append(card)
    return True


def play_pokemon(state: GameState, player_idx: int, hand_index: int, bench: bool = False) -> bool:
    player = state.player1 if player_idx == 0 else state.player2
    assert hand_index < len(player.hand), "Invalid hand index"
    assert not player.pokemon_played_this_turn, "Already played a Pokemon this turn"
    
    card = player.hand[hand_index]
    assert isinstance(card, PokemonCard), "Card must be a Pokemon"
    
    if bench:
        assert len(player.bench) < 5, "Bench is full"
        player.bench.append(PokemonInPlay(card=card))
    else:
        assert player.active_pokemon is None, "Active Pokemon already exists"
        player.active_pokemon = PokemonInPlay(card=card)
    
    player.hand.pop(hand_index)
    player.pokemon_played_this_turn = True
    return True


def attach_energy(state: GameState, player_idx: int, hand_index: int, pokemon_index: Optional[int] = None) -> bool:
    player = state.player1 if player_idx == 0 else state.player2
    assert hand_index < len(player.hand), "Invalid hand index"
    assert player.energy_attached_this_turn < 1, "Already attached energy this turn"
    
    card = player.hand[hand_index]
    assert isinstance(card, EnergyCard), "Card must be an Energy"
    
    target = player.active_pokemon
    if pokemon_index is not None:
        assert pokemon_index < len(player.bench), "Invalid bench index"
        target = player.bench[pokemon_index]
    
    assert target is not None, "No Pokemon to attach energy to"
    
    target.attached_energy.append(card)
    player.hand.pop(hand_index)
    player.energy_attached_this_turn += 1
    return True


def can_attack(pokemon: PokemonInPlay) -> bool:
    if pokemon.status == "asleep" or pokemon.status == "paralyzed":
        return False
    
    attached_types = [e.energy_type for e in pokemon.attached_energy]
    required_types = list(pokemon.card.attack_cost)
    
    for required in required_types:
        if required == EnergyType.COLORLESS:
            continue
        if required not in attached_types:
            return False
        attached_types.remove(required)
    
    return True


def attack(state: GameState, player_idx: int) -> bool:
    player = state.player1 if player_idx == 0 else state.player2
    opponent = state.player2 if player_idx == 0 else state.player1
    
    assert player.active_pokemon is not None, "No active Pokemon"
    assert opponent.active_pokemon is not None, "Opponent has no active Pokemon"
    assert can_attack(player.active_pokemon), "Cannot attack"
    
    damage = player.active_pokemon.card.attack_damage
    opponent.active_pokemon.damage += damage
    
    if opponent.active_pokemon.is_knocked_out:
        take_prize(state, player_idx)
        opponent.active_pokemon = None
    
    return True


def take_prize(state: GameState, player_idx: int) -> bool:
    player = state.player1 if player_idx == 0 else state.player2
    assert len(player.prizes) > 0, "No prizes remaining"
    
    prize = player.prizes.pop()
    player.hand.append(prize)
    
    if len(player.prizes) == 0:
        state.winner = player_idx
    
    return True


def check_win_condition(state: GameState) -> Optional[int]:
    if state.winner is not None:
        return state.winner
    
    if len(state.player1.prizes) == 0:
        state.winner = 0
        return 0
    if len(state.player2.prizes) == 0:
        state.winner = 1
        return 1
    
    if state.player1.active_pokemon is None and len(state.player1.bench) == 0:
        state.winner = 1
        return 1
    if state.player2.active_pokemon is None and len(state.player2.bench) == 0:
        state.winner = 0
        return 0
    
    return None


def end_turn(state: GameState) -> None:
    current_player = state.current_player_state
    current_player.reset_turn_flags()
    
    state.current_player = 1 - state.current_player
    state.turn_number += 1
    
    new_player = state.current_player_state
    draw_card(state, state.current_player)


def get_observable_state(state: GameState, player_idx: int) -> dict:
    player = state.player1 if player_idx == 0 else state.player2
    opponent = state.player2 if player_idx == 0 else state.player1
    
    return {
        "current_player": state.current_player,
        "turn_number": state.turn_number,
        "my_hand_size": len(player.hand),
        "my_deck_size": len(player.deck),
        "my_prizes_remaining": len(player.prizes),
        "my_discard_size": len(player.discard),
        "my_active_pokemon": _pokemon_to_dict(player.active_pokemon) if player.active_pokemon else None,
        "my_bench": [_pokemon_to_dict(p) for p in player.bench],
        "opponent_hand_size": len(opponent.hand),
        "opponent_deck_size": len(opponent.deck),
        "opponent_prizes_remaining": len(opponent.prizes),
        "opponent_active_pokemon": _pokemon_to_dict_public(opponent.active_pokemon) if opponent.active_pokemon else None,
        "opponent_bench_size": len(opponent.bench),
        "winner": state.winner,
    }


def _pokemon_to_dict(pokemon: Optional[PokemonInPlay]) -> Optional[dict]:
    if pokemon is None:
        return None
    return {
        "name": pokemon.card.name,
        "hp": pokemon.card.hp,
        "current_hp": pokemon.card.hp - pokemon.damage,
        "damage": pokemon.damage,
        "energy_types": [e.value for e in pokemon.card.energy_types],
        "attached_energy": [e.energy_type.value for e in pokemon.attached_energy],
        "attack_cost": [e.value for e in pokemon.card.attack_cost],
        "attack_damage": pokemon.card.attack_damage,
        "status": pokemon.status,
    }


def _pokemon_to_dict_public(pokemon: Optional[PokemonInPlay]) -> Optional[dict]:
    if pokemon is None:
        return None
    return {
        "name": pokemon.card.name,
        "hp": pokemon.card.hp,
        "current_hp": pokemon.card.hp - pokemon.damage,
        "damage": pokemon.damage,
        "energy_types": [e.value for e in pokemon.card.energy_types],
        "attached_energy_count": len(pokemon.attached_energy),
        "status": pokemon.status,
    }
