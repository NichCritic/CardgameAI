import numpy as np
from typing import Optional
from game_engine import get_observable_state
from game_state import GameState


def encode_state(state: GameState, player_idx: int) -> np.ndarray:
    obs = get_observable_state(state, player_idx)
    
    features = []
    
    features.append(float(obs["current_player"] == player_idx))
    features.append(float(obs["turn_number"]) / 100.0)
    
    features.append(float(obs["my_hand_size"]) / 60.0)
    features.append(float(obs["my_deck_size"]) / 60.0)
    features.append(float(obs["my_prizes_remaining"]) / 6.0)
    features.append(float(obs["my_discard_size"]) / 60.0)
    
    features.append(float(obs["opponent_hand_size"]) / 60.0)
    features.append(float(obs["opponent_deck_size"]) / 60.0)
    features.append(float(obs["opponent_prizes_remaining"]) / 6.0)
    features.append(float(obs["opponent_bench_size"]) / 5.0)
    
    my_active = obs["my_active_pokemon"]
    if my_active:
        features.append(1.0)
        features.append(float(my_active["current_hp"]) / float(my_active["hp"]))
        features.append(float(my_active["damage"]) / float(my_active["hp"]))
        features.append(float(len(my_active["attached_energy"])) / 10.0)
        features.append(float(my_active["attack_damage"]) / 100.0)
        
        energy_types = [0.0] * 8
        for e in my_active["energy_types"]:
            energy_types[_energy_type_to_idx(e)] = 1.0
        features.extend(energy_types)
        
        attack_cost_types = [0.0] * 8
        for e in my_active["attack_cost"]:
            attack_cost_types[_energy_type_to_idx(e)] += 1.0
        features.extend(attack_cost_types)
    else:
        features.extend([0.0] * 19)
    
    my_bench = obs["my_bench"]
    for i in range(5):
        if i < len(my_bench):
            p = my_bench[i]
            features.append(1.0)
            features.append(float(p["current_hp"]) / float(p["hp"]))
            features.append(float(p["damage"]) / float(p["hp"]))
            features.append(float(len(p["attached_energy"])) / 10.0)
        else:
            features.extend([0.0] * 4)
    
    opponent_active = obs["opponent_active_pokemon"]
    if opponent_active:
        features.append(1.0)
        features.append(float(opponent_active["current_hp"]) / float(opponent_active["hp"]))
        features.append(float(opponent_active["damage"]) / float(opponent_active["hp"]))
        features.append(float(opponent_active["attached_energy_count"]) / 10.0)
        
        energy_types = [0.0] * 8
        for e in opponent_active["energy_types"]:
            energy_types[_energy_type_to_idx(e)] = 1.0
        features.extend(energy_types)
    else:
        features.extend([0.0] * 12)
    
    return np.array(features, dtype=np.float32)


def _energy_type_to_idx(energy_type: str) -> int:
    mapping = {
        "fire": 0,
        "water": 1,
        "grass": 2,
        "electric": 3,
        "psychic": 4,
        "fighting": 5,
        "dark": 6,
        "steel": 7,
        "colorless": 0,
    }
    return mapping.get(energy_type.lower(), 0)
