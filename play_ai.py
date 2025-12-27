from game_engine import initialize_game, check_win_condition
from game import apply_action, get_valid_actions
from state_encoder import encode_state
from action_encoder import ActionEncoder
from dqn_agent import DQNAgent
from actions import Action, ActionType
from train_ai import build_action_space


def play_against_ai(model_path: str = "dqn_model.pt"):
    action_encoder = ActionEncoder()
    build_action_space(action_encoder)
    
    sample_state = initialize_game()
    sample_obs = encode_state(sample_state, 0)
    state_dim = len(sample_obs)
    
    agent = DQNAgent(state_dim, action_encoder)
    agent.load(model_path)
    agent.epsilon = 0.0
    
    state = initialize_game()
    
    print("Playing against AI. You are Player 1, AI is Player 2.")
    print("Type 'help' for commands.\n")
    
    while state.winner is None:
        print(f"\n--- Turn {state.turn_number} ---")
        
        if state.current_player == 0:
            print("Your turn!")
            obs = get_observable_state(state, 0)
            print(f"Hand size: {obs['my_hand_size']}, Prizes: {obs['my_prizes_remaining']}")
            
            if obs['my_active_pokemon']:
                print(f"Active: {obs['my_active_pokemon']['name']} ({obs['my_active_pokemon']['current_hp']}/{obs['my_active_pokemon']['hp']} HP)")
            
            if obs['opponent_active_pokemon']:
                print(f"Opponent Active: {obs['opponent_active_pokemon']['name']} ({obs['opponent_active_pokemon']['current_hp']}/{obs['opponent_active_pokemon']['hp']} HP)")
            
            actions = get_valid_actions(state)
            print(f"\nValid actions:")
            for i, action in enumerate(actions):
                print(f"  {i}: {action.action_type.value}", end="")
                if action.hand_index is not None:
                    print(f" (hand_index={action.hand_index})", end="")
                if action.pokemon_index is not None:
                    print(f" (pokemon_index={action.pokemon_index})", end="")
                if action.bench:
                    print(f" (bench=True)", end="")
                print()
            
            try:
                choice = input("\nChoose action (number) or 'auto': ")
                if choice == 'auto':
                    action = agent.select_action(state, 0, training=False)
                    print(f"AI chose: {action.action_type.value}")
                elif choice == 'help':
                    print("Commands: 'auto' - let AI play, 'help' - show this, number - choose action")
                    continue
                else:
                    action_idx = int(choice)
                    if 0 <= action_idx < len(actions):
                        action = actions[action_idx]
                    else:
                        print("Invalid choice")
                        continue
            except (ValueError, KeyboardInterrupt):
                print("Exiting...")
                return
            
            apply_action(state, action)
        else:
            print("AI's turn...")
            action = agent.select_action(state, 1, training=False)
            print(f"AI chose: {action.action_type.value}")
            apply_action(state, action)
        
        check_win_condition(state)
    
    if state.winner is not None:
        print(f"\nPlayer {state.winner + 1} wins!")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "dqn_model.pt"
    play_against_ai(model_path)
