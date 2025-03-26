import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent import Agent
from game.state import State
from game.action import Action
from game.card import Card
import random

# neural network to select the action to take
class ActionNN(nn.Module):
    def __init__(self, input_size: int = 40 * 4 + 2, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output: DRAW, DRAW_DISCARD, CALL
        self.fc3 = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

# neural network that takes the input cards and outputs the position to swap
class SwapNN(nn.Module):
    def __init__(self, input_size: int = 40 * 4, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output: which card to swap (0, 1, 2) or to not swap (3)
        self.fc3 = nn.Linear(hidden_size, 4)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

class AIAgent(Agent):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.action_model = ActionNN()
        self.swap_model = SwapNN()
        self.action_optimizer = torch.optim.Adam(self.action_model.parameters(), lr=0.001)
        self.swap_optimizer = torch.optim.Adam(self.swap_model.parameters(), lr=0.001)
        self.epsilon = epsilon
    
    def _encode_basic_state(self, state: State) -> torch.Tensor:
        return torch.cat((torch.tensor([state.first_turn, state.called], dtype=torch.bool), self._encode_cards(state.hands[state.turn] + [state.discard[-1]])))

    def _encode_swap_state(self, state: State, drawn_card: Card) -> torch.Tensor:
        return self._encode_cards(state.hands[state.turn] + [drawn_card])
    
    # each cards is represented as a 40 bit value
    def _encode_cards(self, cards: list[Card]) -> torch.Tensor:
        encoding = torch.zeros(len(cards) * 40, dtype=torch.bool)
        
        for i, card in enumerate(cards):
            idx: int = card.val - 2
            idx += card.suit.value * 10

            encoding[i * 40 + idx] = 1
        
        return encoding

    def take_turn(self, state: State) -> tuple[Action, int]:
        if random.random() < self.epsilon:
            return self._random_action(state)
        
        with torch.no_grad():
            state_tensor = self._encode_basic_state(state)
            action_values = self.action_model(state_tensor)
            
            # mask invalid action
            # can't call if already called
            if state.called:
                action_values[Action.CALL.value] = float('-inf')
            
            action = Action(torch.argmax(action_values).item())
            
            # If drawing from deck, we need to make a second decision
            if action == Action.DRAW:
                discard_state = self._encode_swap_state(state, state.deck[-1])
                position_values = self.swap_model(discard_state)
                position = torch.argmax(position_values).item()
                return (action, position)
            elif action == Action.DRAW_DISCARD:
                discard_state = self._encode_swap_state(state, state.discard[-1])
                position_values = self.swap_model(discard_state)
                # mask invalid action
                # can't discard the drawn card from the discard pile
                position_values[3] = float('-inf')
                position = torch.argmax(position_values).item()
                return (action, position)
            else:  # CALL
                return (action, 0)

    # first turn is handled in the state encoding
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)

    # action masking is already handled in take_turn
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)
    
    def _random_action(self, state: State) -> tuple[Action, int]:
        if state.called:
            action = random.choice([Action.DRAW, Action.DRAW_DISCARD])
        else:
            action = random.choice([Action.DRAW, Action.DRAW_DISCARD, Action.CALL])
        return (action, random.randint(0, 2))

    # TODO: idk this training method b/c I'm not sure about to give rewards based on individual actions and not the entire game
    # what is here is just ai crap
    def train_action(self, state: State, action: Action, reward: float):
        """Train the action selection network"""
        self.action_optimizer.zero_grad()
        state_tensor = self._encode_basic_state(state)
        action_values = self.action_model(state_tensor)
        
        target = torch.tensor([reward], dtype=torch.float32)
        loss = F.mse_loss(action_values[action.value], target)
        loss.backward()
        self.action_optimizer.step()

    def train_swap(self, state: State, drawn_card: Card, position: int, reward: float):
        """Train the card swapping network"""
        self.swap_optimizer.zero_grad()
        state_tensor = self._encode_swap_state(state, drawn_card)
        position_values = self.swap_model(state_tensor)
        
        target = torch.tensor([reward], dtype=torch.float32)
        loss = F.mse_loss(position_values[position], target)
        loss.backward()
        self.swap_optimizer.step()