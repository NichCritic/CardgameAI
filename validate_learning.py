import copy
from typing import Optional
from game_engine import initialize_game, check_win_condition
from game import apply_action, get_valid_actions
from game_state import GameState
from state_encoder import encode_state
from action_encoder import ActionEncoder
from dqn_agent import DQNAgent
from actions import Action, ActionType
from train_ai import play_game, build_action_space, random_action


def play_game_vs_random(agent: DQNAgent, training: bool = False):
    state = initialize_game()
    initial_state = copy.deepcopy(state)
    prev_states = {0: initial_state, 1: initial_state}
    actions_taken = {0: [], 1: []}
    
    max_turns = 200
    turn_count = 0
    
    while state.winner is None and turn_count < max_turns:
        player_idx = state.current_player
        prev_state = prev_states[player_idx]
        
        if player_idx == 0:
            action = agent.select_action(state, player_idx, training=training)
        else:
            action = random_action(state)
        
        state_before_action = copy.deepcopy(state)
        apply_action(state, action)
        check_win_condition(state)
        
        done = state.winner is not None
        actions_taken[player_idx].append(action)
        
        if training and prev_state is not None and len(actions_taken[player_idx]) > 0:
            from train_ai import calculate_reward
            reward = calculate_reward(state, prev_state, player_idx, done)
            if player_idx == 0:
                agent.store_transition(prev_state, actions_taken[player_idx][-1], reward, state, done, player_idx)
        
        if not done:
            prev_states[player_idx] = state_before_action
        
        if state.current_player != player_idx:
            turn_count += 1
    
    return state.winner, turn_count


def evaluate_agent(agent: DQNAgent, num_games: int = 100, opponent_random: bool = True) -> float:
    wins = 0
    for _ in range(num_games):
        if opponent_random:
            winner, _ = play_game_vs_random(agent, training=False)
        else:
            winner, _ = play_game(agent, opponent_agent=agent, training=False)
        if winner == 0:
            wins += 1
    return wins / num_games


def validate_learning(model_path: str = "dqn_model.pt", num_test_games: int = 100):
    action_encoder = ActionEncoder()
    build_action_space(action_encoder, num_games=50)
    
    sample_state = initialize_game()
    sample_obs = encode_state(sample_state, 0)
    state_dim = len(sample_obs)
    
    trained_agent = DQNAgent(state_dim, action_encoder)
    trained_agent.load(model_path)
    trained_agent.epsilon = 0.0
    
    untrained_agent = DQNAgent(state_dim, action_encoder)
    untrained_agent.epsilon = 0.0
    
    print("Evaluating trained agent vs random opponent...")
    trained_vs_random = evaluate_agent(trained_agent, num_test_games, opponent_random=True)
    
    print("Evaluating untrained agent vs random opponent...")
    untrained_vs_random = evaluate_agent(untrained_agent, num_test_games, opponent_random=True)
    
    print("Evaluating trained agent vs untrained agent...")
    trained_wins = 0
    for _ in range(num_test_games):
        winner, _ = play_game(trained_agent, opponent_agent=untrained_agent, training=False)
        if winner == 0:
            trained_wins += 1
    trained_vs_untrained = trained_wins / num_test_games
    
    print("\n" + "="*60)
    print("LEARNING VALIDATION RESULTS")
    print("="*60)
    print(f"Trained agent vs Random:     {trained_vs_random:.2%} win rate")
    print(f"Untrained agent vs Random:  {untrained_vs_random:.2%} win rate")
    print(f"Trained vs Untrained:       {trained_vs_untrained:.2%} win rate")
    print("="*60)
    
    learning_detected = False
    issues = []
    
    if trained_vs_random > 0.55:
        print("✓ Trained agent performs better than random (>55%)")
        learning_detected = True
    else:
        issues.append(f"Trained agent win rate ({trained_vs_random:.2%}) is not significantly above random")
    
    if trained_vs_untrained > 0.55:
        print("✓ Trained agent beats untrained agent (>55%)")
        learning_detected = True
    else:
        issues.append(f"Trained agent does not consistently beat untrained agent ({trained_vs_untrained:.2%})")
    
    if trained_vs_random > untrained_vs_random + 0.05:
        print("✓ Trained agent shows improvement over untrained agent")
        learning_detected = True
    else:
        issues.append(f"Trained agent improvement ({trained_vs_random - untrained_vs_random:.2%}) is minimal")
    
    if learning_detected:
        print("\n✓ LEARNING VALIDATION PASSED")
    else:
        print("\n✗ LEARNING VALIDATION FAILED")
        print("\nIssues detected:")
        for issue in issues:
            print(f"  - {issue}")
    
    return learning_detected


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "dqn_model.pt"
    validate_learning(model_path)