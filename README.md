# Pokemon TCG Game Engine

A simplified Pokemon Trading Card Game engine designed for machine learning training.

## Features

- **Card Types**: Pokemon, Energy, and Trainer cards
- **Hidden Information**: Hands and decks are hidden from opponents
- **Random Elements**: Card draws, deck shuffling
- **Core Mechanics**: 
  - Drawing cards
  - Playing Pokemon to active or bench
  - Attaching Energy cards
  - Attacking opponent Pokemon
  - Taking prizes when knocking out Pokemon
  - Win conditions (take all prizes or opponent has no Pokemon)

## Game Rules (Simplified)

1. Each player starts with 7 cards and 6 prizes
2. Players take turns drawing, playing cards, and attacking
3. One Pokemon can be played per turn
4. One Energy can be attached per turn
5. Pokemon can attack if they have the required Energy attached
6. When a Pokemon is knocked out, the attacker takes a prize
7. Win by taking all 6 prizes or if opponent has no Pokemon left

## Usage

```python
from game import initialize_game, apply_action, get_valid_actions, get_observable_state
from actions import Action, ActionType

# Initialize a game
state = initialize_game()

# Get observable state for current player
obs = get_observable_state(state, state.current_player)

# Get valid actions
actions = get_valid_actions(state)

# Apply an action
action = actions[0]
apply_action(state, action)

# Check win condition
winner = state.winner
```

## Structure

- `cards.py` - Card data structures
- `game_state.py` - Game state data structures
- `game_engine.py` - Core game mechanics
- `actions.py` - Action definitions for ML
- `game.py` - High-level game interface

## ML Training

The engine provides:
- `get_observable_state()` - Returns observable state for a player (hides opponent's hand)
- `get_valid_actions()` - Returns all valid actions for current player
- `apply_action()` - Applies an action and updates game state
- `check_win_condition()` - Checks if game is over
