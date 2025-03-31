import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent import Agent
from game.state import State
from game.action import Action
from game.card import Card
import random

from util import probability

# neural network to select the action to take
class ActionNN(nn.Module):
    def __init__(self, input_size: int = 40 * 4 + 4 + 2, hidden_size: int = 128 * 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Output: DRAW, DRAW_DISCARD, CALL
        self.fc4 = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

# neural network that takes the input cards and outputs the position to swap
class SwapNN(nn.Module):
    def __init__(self, input_size: int = 40 * 4 + 4, hidden_size: int = 128 * 2):
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.fc2: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.fc3: nn.Linear = nn.Linear(hidden_size, hidden_size)
        # Output: which card to swap (0, 1, 2) or to not swap (3)
        self.fc4: nn.Linear = nn.Linear(hidden_size, 4)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

class AIAgent(Agent):
    # TODO: fix adding in the models as parameters
    def __init__(self, action_nn: ActionNN = ActionNN(), swap_nn: SwapNN = SwapNN(), epsilon: float = 0.05):
        super().__init__()
        self.action_model: ActionNN = action_nn
        self.swap_model: SwapNN = swap_nn
        self.action_optimizer: torch.optim.Adam = torch.optim.Adam(self.action_model.parameters(), lr=0.001)
        self.swap_optimizer: torch.optim.Adam = torch.optim.Adam(self.swap_model.parameters(), lr=0.001)
        self.epsilon: float = epsilon
        self.experiences: list[dict[str, torch.Tensor | Action | int | None]] = []

    # save the model weights and optimizer states to a file
    def save_checkpoint(self, path: str):
        checkpoint = {
            'action_model': self.action_model.state_dict(),
            'swap_model': self.swap_model.state_dict(),
            'action_optimizer': self.action_optimizer.state_dict(),
            'swap_optimizer': self.swap_optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)

    # load the model weights and optimizer states from a file
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.action_model.load_state_dict(checkpoint['action_model'])
        self.swap_model.load_state_dict(checkpoint['swap_model'])
        self.action_optimizer.load_state_dict(checkpoint['action_optimizer'])
        self.swap_optimizer.load_state_dict(checkpoint['swap_optimizer'])
        self.epsilon = checkpoint['epsilon']
    
    def _encode_basic_state(self, state: State) -> torch.Tensor:
        return torch.cat((
            self._encode_cards(state.hands[state.turn] + [state.discard[-1]]),
            torch.tensor(probability.expected_values(state), dtype=torch.float),
            torch.tensor([state.first_turn, state.called], dtype=torch.float) 
        ))

    def _encode_swap_state(self, state: State, drawn_card: Card) -> torch.Tensor:
        return torch.cat((
            self._encode_cards(state.hands[state.turn] + [drawn_card]),
            torch.tensor(probability.expected_values(state), dtype=torch.float)
        ))
    
    # each cards is represented as a 40 bit value
    def _encode_cards(self, cards: list[Card]) -> torch.Tensor:
        encoding = torch.zeros(len(cards) * 40, dtype=torch.float)
        
        for i, card in enumerate(cards):
            idx: int = card.val - 2
            idx += card.suit.value * 10

            encoding[i * 40 + idx] = 1.0
        
        return encoding

    def take_turn(self, state: State) -> tuple[Action, int]:
        target_action: tuple[Action, int] = None

        state_tensor: torch.Tensor = self._encode_basic_state(state)
        swap_tensor: torch.Tensor | None = None

        if random.random() < self.epsilon:
            # don't save experience for random actions TODO: do I wan't to do this?
            return self._random_action(state)
        
        with torch.no_grad():
            action_values: torch.Tensor = self.action_model(state_tensor)
            
            # mask invalid action
            # can't call if already called
            if state.called != -1:
                action_values[Action.CALL.value] = float('-inf')

            action: Action = Action(torch.argmax(action_values).item())
            
            # If drawing from deck, we need to make a second decision
            if action == Action.DRAW:
                swap_tensor = self._encode_swap_state(state, state.deck[-1])
                position_values = self.swap_model(swap_tensor)
                position = torch.argmax(position_values).item()
                target_action = (action, position)
            elif action == Action.DRAW_DISCARD:
                swap_tensor = self._encode_swap_state(state, state.discard[-1])
                position_values = self.swap_model(swap_tensor)
                # mask invalid action
                # can't discard the drawn card from the discard pile
                position_values[3] = float('-inf')
                position = torch.argmax(position_values).item()
                target_action = (action, position)
            else: # CALL
                target_action = (action, 0)
        
        assert target_action is not None
        self.store_experience(state_tensor, target_action, swap_tensor)

        return target_action

    # first turn is handled in the state encoding
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)

    # action masking is already handled in take_turn
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)
    
    def _random_action(self, state: State) -> tuple[Action, int]:
        if state.called != -1:
            action = random.choice([Action.DRAW, Action.DRAW_DISCARD])
        else:
            action = random.choice([Action.DRAW, Action.DRAW_DISCARD, Action.CALL])

        idx = 0        
        if action == Action.DRAW:
            idx = random.randint(0, 3)
        elif action == Action.DRAW_DISCARD:
            idx = random.randint(0, 2)
        
        return (action, idx)

    # TODO: implement training

    def flush_experience(self):
        self.experiences = []

    # stores a state experience for training after a game
    def store_experience(self, state_tensor: torch.Tensor, action: tuple[Action, int], swap_tensor: torch.Tensor | None):
        self.experiences.append({
            'state': state_tensor,
            'swap_state': swap_tensor,
            'action': action[0],
            'position': action[1]
        })

    # trains on an entire game based on the reward
    # flushes the experience buffer
    def train(self, reward: float):
        #print("Starting training")
        # Skip if no experiences to train on
        if not self.experiences:
            return
        
        # Calculate discounted rewards for each step
        gamma: float = 0.99  # discount factor
        discounted_rewards: list[float] = []
        running_reward: float = reward
        for _ in range(len(self.experiences)):
            discounted_rewards.insert(0, running_reward)
            running_reward *= gamma

        # Convert rewards to tensor
        rewards: torch.Tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        
        # Train action network
        action_states: torch.Tensor = torch.stack([exp['state'] for exp in self.experiences])
        action_outputs: torch.Tensor = self.action_model(action_states)
        action_targets: torch.Tensor = torch.zeros_like(action_outputs)
        for i, exp in enumerate(self.experiences):
            action_targets[i][exp['action'].value] = rewards[i]
        
        action_loss: torch.Tensor = F.mse_loss(action_outputs, action_targets)
        self.action_optimizer.zero_grad()
        action_loss.backward()
        self.action_optimizer.step()
        
        # Train swap network
        # Filter only for experiences that involved drawing
        swap_experiences: list[tuple[torch.Tensor, int, float]] = [
            (exp['swap_state'], exp['position'], reward)
            for exp, reward in zip(self.experiences, discounted_rewards)
            if exp['swap_state'] is not None and isinstance(exp['action'], Action)
        ]
        
        if swap_experiences:
            swap_states = torch.stack([exp[0] for exp in swap_experiences])
            swap_outputs = self.swap_model(swap_states)
            swap_targets = torch.zeros_like(swap_outputs)
            for i, exp in enumerate(swap_experiences):
                swap_targets[i][exp[1]] = exp[2]
            
            swap_loss = F.mse_loss(swap_outputs, swap_targets)
            self.swap_optimizer.zero_grad()
            swap_loss.backward()
            self.swap_optimizer.step()
        
        #print("Training complete")
        # Clear experiences after training
        self.flush_experience()