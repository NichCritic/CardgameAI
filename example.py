from game import initialize_game, apply_action, get_valid_actions, get_observable_state
from game_engine import check_win_condition
from actions import Action, ActionType


def play_random_game():
    import random
    
    state = initialize_game()
    actions_this_turn = 0
    
    while state.winner is None:
        turn_start = state.turn_number
        print(f"\n--- Turn {state.turn_number} - Player {state.current_player + 1} ---")
        
        obs = get_observable_state(state, state.current_player)
        print(f"Hand size: {obs['my_hand_size']}, Prizes: {obs['my_prizes_remaining']}")
        
        if obs['my_active_pokemon']:
            print(f"Active: {obs['my_active_pokemon']['name']} ({obs['my_active_pokemon']['current_hp']}/{obs['my_active_pokemon']['hp']} HP)")
        else:
            print("No active Pokemon")
        
        if obs['opponent_active_pokemon']:
            print(f"Opponent Active: {obs['opponent_active_pokemon']['name']} ({obs['opponent_active_pokemon']['current_hp']}/{obs['opponent_active_pokemon']['hp']} HP)")
        
        actions = get_valid_actions(state)
        
        if not actions:
            break
        
        action = None
        for a in actions:
            if a.action_type == ActionType.ATTACK:
                action = a
                print("Attacking!")
                break
        
        if action is None:
            for a in actions:
                if a.action_type == ActionType.PLAY_POKEMON and obs['my_active_pokemon'] is None:
                    action = a
                    print(f"Playing Pokemon from hand index {a.hand_index}")
                    break
        
        if action is None:
            for a in actions:
                if a.action_type == ActionType.ATTACH_ENERGY and obs['my_active_pokemon'] is not None:
                    action = a
                    print(f"Attaching Energy from hand index {a.hand_index}")
                    break
        
        if action is None:
            action = Action(ActionType.END_TURN)
            print("Ending turn")
        
        apply_action(state, action)
        check_win_condition(state)
        
        if state.turn_number != turn_start:
            actions_this_turn = 0
        else:
            actions_this_turn += 1
            if actions_this_turn > 10:
                apply_action(state, Action(ActionType.END_TURN))
                actions_this_turn = 0
        
        if state.turn_number > 100:
            print("Game too long, stopping")
            break
    
    if state.winner is not None:
        print(f"\nPlayer {state.winner + 1} wins!")
    else:
        print("\nGame ended without winner")


if __name__ == "__main__":
    play_random_game()
