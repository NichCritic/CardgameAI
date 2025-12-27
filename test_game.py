from game import initialize_game, apply_action, get_valid_actions, get_observable_state
from game_engine import check_win_condition
from actions import Action, ActionType


def test_basic_gameplay():
    state = initialize_game()
    
    assert state.current_player == 0
    assert len(state.player1.hand) == 7
    assert len(state.player2.hand) == 7
    assert len(state.player1.prizes) == 6
    assert len(state.player2.prizes) == 6
    
    actions = get_valid_actions(state)
    assert len(actions) > 0
    
    play_pokemon_action = None
    for a in actions:
        if a.action_type == ActionType.PLAY_POKEMON:
            play_pokemon_action = a
            break
    
    if play_pokemon_action:
        apply_action(state, play_pokemon_action)
        assert state.player1.active_pokemon is not None
    
    apply_action(state, Action(ActionType.END_TURN))
    assert state.current_player == 1
    assert state.turn_number == 2
    
    print("Basic gameplay test passed!")


if __name__ == "__main__":
    test_basic_gameplay()
