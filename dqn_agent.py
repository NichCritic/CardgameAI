import random
import numpy as np
import torch
import torch.optim as optim
from typing import Optional
from game_state import GameState
from state_encoder import encode_state
from action_encoder import ActionEncoder
from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer, Transition
from actions import Action


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_encoder: ActionEncoder,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: Optional[torch.device] = None,
    ):
        self.action_encoder = action_encoder
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.q_network = DQNNetwork(state_dim, action_encoder.get_max_actions()).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_encoder.get_max_actions()).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state: GameState, player_idx: int, training: bool = True) -> Action:
        if training and random.random() < self.epsilon:
            valid_actions = self.action_encoder.get_valid_action_indices(state)
            if not valid_actions:
                return Action(ActionType.END_TURN)
            action_idx = random.choice(valid_actions)
            return self.action_encoder.decode(action_idx)
        
        state_tensor = torch.FloatTensor(encode_state(state, player_idx)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        action_mask = self.action_encoder.get_action_mask(state, max_size=q_values.shape[1])
        if not any(action_mask):
            return Action(ActionType.END_TURN)
        
        q_values_np = q_values.cpu().numpy()[0]
        masked_q_values = np.where(action_mask, q_values_np, -np.inf)
        action_idx = np.argmax(masked_q_values)
        
        action = self.action_encoder.decode(action_idx)
        if action is None:
            valid_actions = self.action_encoder.get_valid_action_indices(state)
            if valid_actions:
                action_idx = random.choice(valid_actions)
                action = self.action_encoder.decode(action_idx)
            else:
                action = Action(ActionType.END_TURN)
        
        return action
    
    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def store_transition(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: GameState,
        done: bool,
        player_idx: int,
    ) -> None:
        state_vec = encode_state(state, player_idx)
        next_state_vec = encode_state(next_state, player_idx)
        action_idx = self.action_encoder.encode(action)
        action_mask = self.action_encoder.get_action_mask(state)
        next_action_mask = self.action_encoder.get_action_mask(next_state)
        
        transition = Transition(
            state=state_vec,
            action=action_idx,
            reward=reward,
            next_state=next_state_vec,
            done=done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )
        self.replay_buffer.push(transition)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        if len(self.replay_buffer) < batch_size:
            return None
        
        batch = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.BoolTensor([t.done for t in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            for i, t in enumerate(batch):
                mask_tensor = torch.tensor(t.next_action_mask[:next_q_values.shape[1]], dtype=torch.bool).to(self.device)
                if len(mask_tensor) < next_q_values.shape[1]:
                    padding = torch.zeros(next_q_values.shape[1] - len(mask_tensor), dtype=torch.bool).to(self.device)
                    mask_tensor = torch.cat([mask_tensor, padding])
                next_q_values[i] = torch.where(
                    mask_tensor,
                    next_q_values[i],
                    torch.tensor(-np.inf, device=self.device)
                )
            next_max_q = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * next_max_q * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str) -> None:
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'action_encoder': self.action_encoder,
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.action_encoder = checkpoint['action_encoder']
