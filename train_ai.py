import random
import copy
from typing import Optional
from game_engine import initialize_game, check_win_condition
from game import apply_action, get_valid_actions
from game_state import GameState
from state_encoder import encode_state
from action_encoder import ActionEncoder
from dqn_agent import DQNAgent
from actions import Action, ActionType


def calculate_reward(state, prev_state, player_idx, done):
    if done:
        if state.winner == player_idx:
            return 100.0
        elif state.winner == 1 - player_idx:
            return -100.0
        else:
            return 0.0
    
    reward = 0.0
    
    current_player = state.current_player_state
    opponent = state.opponent_player_state
    
    prev_current_player = prev_state.current_player_state
    prev_opponent = prev_state.opponent_player_state
    
    if player_idx == state.current_player:
        reward += (len(prev_current_player.prizes) - len(current_player.prizes)) * 10.0
        reward -= (len(prev_opponent.prizes) - len(opponent.prizes)) * 10.0
        
        if current_player.active_pokemon and prev_current_player.active_pokemon:
            prev_damage = prev_current_player.active_pokemon.damage
            curr_damage = current_player.active_pokemon.damage
            if curr_damage < prev_damage:
                reward += 5.0
        
        if opponent.active_pokemon and prev_opponent.active_pokemon:
            prev_damage = prev_opponent.active_pokemon.damage
            curr_damage = opponent.active_pokemon.damage
            if curr_damage > prev_damage:
                reward += 5.0
    
    return reward


def random_action(state):
    actions = get_valid_actions(state)
    if not actions:
        return Action(ActionType.END_TURN)
    return random.choice(actions)


def play_game(agent: DQNAgent, opponent_agent: Optional[DQNAgent] = None, training: bool = True):
    state = initialize_game()
    initial_state = copy.deepcopy(state)
    prev_states = {0: initial_state, 1: initial_state}
    actions_taken = {0: [], 1: []}
    
    max_turns = 200
    turn_count = 0
    
    while state.winner is None and turn_count < max_turns:
        player_idx = state.current_player
        prev_state = prev_states[player_idx]
        
        if opponent_agent is None or player_idx == 0:
            action = agent.select_action(state, player_idx, training=training)
        else:
            action = opponent_agent.select_action(state, player_idx, training=training)
        
        state_before_action = copy.deepcopy(state)
        apply_action(state, action)
        check_win_condition(state)
        
        done = state.winner is not None
        
        actions_taken[player_idx].append(action)
        
        if training and prev_state is not None and len(actions_taken[player_idx]) > 0:
            reward = calculate_reward(state, prev_state, player_idx, done)
            
            if player_idx == 0:
                agent.store_transition(prev_state, actions_taken[player_idx][-1], reward, state, done, player_idx)
            elif opponent_agent is not None:
                opponent_agent.store_transition(prev_state, actions_taken[player_idx][-1], reward, state, done, player_idx)
        
        if not done:
            prev_states[player_idx] = state_before_action
        
        if state.current_player != player_idx:
            turn_count += 1
    
    return state.winner, turn_count


def build_action_space(action_encoder: ActionEncoder, num_games: int = 20) -> None:
    for _ in range(num_games):
        state = initialize_game()
        max_turns = 50
        turn_count = 0
        
        while state.winner is None and turn_count < max_turns:
            actions = get_valid_actions(state)
            for action in actions:
                action_encoder.encode(action)
            
            if not actions:
                break
            
            apply_action(state, random.choice(actions))
            check_win_condition(state)
            
            if state.current_player != (turn_count % 2):
                turn_count += 1


def train_agent(
    episodes: int = 10000,
    target_update_freq: int = 100,
    train_freq: int = 4,
    batch_size: int = 32,
    save_freq: int = 1000,
    save_path: str = "dqn_model.pt",
):
    action_encoder = ActionEncoder()
    print("Building action space...")
    build_action_space(action_encoder, num_games=50)
    print(f"Action space size: {action_encoder.get_max_actions()}")
    
    sample_state = initialize_game()
    sample_obs = encode_state(sample_state, 0)
    state_dim = len(sample_obs)
    
    agent = DQNAgent(state_dim, action_encoder)
    
    wins = 0
    total_rewards = []
    episode_wins = []
    
    for episode in range(episodes):
        winner, turns = play_game(agent, training=True)
        
        episode_wins.append(1 if winner == 0 else 0)
        if len(episode_wins) > 100:
            episode_wins.pop(0)
        
        if episode % train_freq == 0 and len(agent.replay_buffer) >= batch_size:
            loss = agent.train_step(batch_size)
            if loss is not None:
                total_rewards.append(loss)
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        agent.update_epsilon()
        
        if episode % 100 == 0:
            win_rate = sum(episode_wins) / len(episode_wins) if episode_wins else 0.0
            avg_loss = sum(total_rewards[-100:]) / len(total_rewards[-100:]) if total_rewards else 0.0
            loss_trend = ""
            if len(total_rewards) >= 200:
                early_loss = sum(total_rewards[:100]) / 100
                recent_loss = sum(total_rewards[-100:]) / 100
                if recent_loss < early_loss * 0.9:
                    loss_trend = " (↓)"
                elif recent_loss > early_loss * 1.1:
                    loss_trend = " (↑)"
            print(f"Episode {episode}, Win Rate: {win_rate:.2f}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}{loss_trend}")
        
        if episode % save_freq == 0 and episode > 0:
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    agent.save(save_path)
    print(f"Training complete. Final model saved to {save_path}")


if __name__ == "__main__":
    from typing import Optional
    train_agent(episodes=10000)
