from typing import Optional
from actions import Action, ActionType
from game_state import GameState
from game import get_valid_actions


class ActionEncoder:
    def __init__(self):
        self.action_to_idx: dict[Action, int] = {}
        self.idx_to_action: dict[int, Action] = {}
        self.next_idx = 0
    
    def encode(self, action: Action) -> int:
        if action not in self.action_to_idx:
            self.action_to_idx[action] = self.next_idx
            self.idx_to_action[self.next_idx] = action
            self.next_idx += 1
        return self.action_to_idx[action]
    
    def decode(self, idx: int) -> Optional[Action]:
        return self.idx_to_action.get(idx)
    
    def get_action_mask(self, state: GameState, max_size: Optional[int] = None) -> list[bool]:
        valid_actions = get_valid_actions(state)
        valid_set = set(valid_actions)
        
        size = max_size if max_size is not None else self.next_idx
        mask = []
        for idx in range(size):
            if idx < self.next_idx:
                action = self.idx_to_action[idx]
                mask.append(action in valid_set)
            else:
                mask.append(False)
        return mask
    
    def get_valid_action_indices(self, state: GameState) -> list[int]:
        valid_actions = get_valid_actions(state)
        return [self.encode(action) for action in valid_actions]
    
    def get_max_actions(self) -> int:
        return max(self.next_idx, 1)
