from game_engine import GameState, initialize_game, draw_card, play_pokemon, attach_energy, attack, end_turn, check_win_condition, get_observable_state
from actions import Action, ActionType


def apply_action(state: GameState, action: Action) -> bool:
    assert state.winner is None, "Game is over"
    assert state.current_player == 0 or state.current_player == 1, "Invalid current player"
    
    if action.action_type == ActionType.DRAW_CARD:
        return draw_card(state, state.current_player)
    
    elif action.action_type == ActionType.PLAY_POKEMON:
        assert action.hand_index is not None, "hand_index required for PLAY_POKEMON"
        return play_pokemon(state, state.current_player, action.hand_index, action.bench)
    
    elif action.action_type == ActionType.ATTACH_ENERGY:
        assert action.hand_index is not None, "hand_index required for ATTACH_ENERGY"
        return attach_energy(state, state.current_player, action.hand_index, action.pokemon_index)
    
    elif action.action_type == ActionType.ATTACK:
        return attack(state, state.current_player)
    
    elif action.action_type == ActionType.END_TURN:
        end_turn(state)
        return True
    
    elif action.action_type == ActionType.PASS:
        return True
    
    return False


def get_valid_actions(state: GameState) -> list[Action]:
    from cards import PokemonCard, EnergyCard
    
    player = state.current_player_state
    
    actions = [Action(ActionType.END_TURN)]
    
    for i, card in enumerate(player.hand):
        if isinstance(card, PokemonCard):
            if player.active_pokemon is None and not player.pokemon_played_this_turn:
                actions.append(Action(ActionType.PLAY_POKEMON, hand_index=i, bench=False))
            if len(player.bench) < 5 and not player.pokemon_played_this_turn:
                actions.append(Action(ActionType.PLAY_POKEMON, hand_index=i, bench=True))
        elif isinstance(card, EnergyCard):
            if player.energy_attached_this_turn < 1:
                if player.active_pokemon is not None:
                    actions.append(Action(ActionType.ATTACH_ENERGY, hand_index=i))
                for bench_idx in range(len(player.bench)):
                    actions.append(Action(ActionType.ATTACH_ENERGY, hand_index=i, pokemon_index=bench_idx))
    
    if player.active_pokemon is not None:
        from game_engine import can_attack
        if can_attack(player.active_pokemon):
            actions.append(Action(ActionType.ATTACK))
    
    return actions
