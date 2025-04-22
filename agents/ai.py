import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent import Agent
from game.state import State
from game.action import Action
from game.card import Card
import random

from util import probability

class TrainingMetrics:
    def __init__(self):
        self.action_losses: list[float] = []
        self.swap_losses: list[float] = []
        self.rewards: list[float] = []
        self.episode_lengths: list[int] = []
        
    def add_metrics(self, action_loss: float | None, swap_loss: float | None, reward: float, episode_length: int):
        if action_loss is not None:
            self.action_losses.append(action_loss.detach().item())
        else:
            self.action_losses.append(0.0)
        
        if swap_loss is not None:
            self.swap_losses.append(swap_loss.detach().item())
        else:
            self.swap_losses.append(0.0)
        
        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)
    
    def plot_metrics(self):
        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
        fig.suptitle('Training Metrics')

        # Plot action losses
        if self.action_losses:
            ax1.plot(self.action_losses, label='Action Loss')
            ax1.set_ylabel('Loss')
            ax1.set_title('Action Network Loss')
            ax1.grid(True)
            ax1.legend()

        # Plot swap losses 
        if self.swap_losses:
            ax2.plot(self.swap_losses, label='Swap Loss')
            ax2.set_ylabel('Loss')
            ax2.set_title('Swap Network Loss')
            ax2.grid(True)
            ax2.legend()

        # Plot rewards
        if self.rewards:
            ax3.plot(self.rewards, label='Reward')
            ax3.set_ylabel('Reward')
            ax3.set_title('Episode Rewards')
            ax3.grid(True)
            ax3.legend()

        # Plot episode lengths
        if self.episode_lengths:
            ax4.plot(self.episode_lengths, label='Length')
            ax4.set_ylabel('Steps')
            ax4.set_title('Episode Lengths')
            ax4.grid(True)
            ax4.legend()

        plt.tight_layout()
        plt.show()

# neural network to select the action to take
class ActionNN(nn.Module):
    def __init__(self, input_size: int = 4 * 4 + 4 + 2 + 4 * 6, hidden_size: int = 256):
        # input is 
        # 3 hand cards
        # 1 top discard card
        # 4 expected values (1 per suit)
        # if it is the first turn
        # if it was called
        # the player to the left (that you have seen before)
        # the player to the right (that you have seen before)
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        # Output: DRAW, DRAW_DISCARD, CALL
        self.fc5 = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return self.fc5(x)

# neural network that takes the input cards and outputs the position to swap
class SwapNN(nn.Module):
    def __init__(self, input_size: int = 4 * 4 + 4, hidden_size: int = 128):
        # input is 3 hand cards, 1 drawn card, 4 expected values (1 per suit)
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
    def __init__(self, epsilon: float = 0.0):
        super().__init__()
        self.action_model: ActionNN = ActionNN()
        self.swap_model: SwapNN = SwapNN()
        self.action_optimizer: torch.optim.Adam = torch.optim.Adam(self.action_model.parameters(), lr=0.0003)
        self.swap_optimizer: torch.optim.Adam = torch.optim.Adam(self.swap_model.parameters(), lr=0.0003)
        self.epsilon: float = epsilon
        self.experiences: list[dict[str, torch.Tensor | Action | int | None]] = []

        self.metrics: TrainingMetrics = TrainingMetrics()

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
            torch.tensor(probability.expected_values(state, see_hand=True), dtype=torch.float),
            torch.tensor([state.first_turn, state.called], dtype=torch.float),
            self._encode_cards(state.hands[(state.turn + len(state.hands) + 1) % len(state.hands)], visible=False), # encode the person to the left (the next player)
            self._encode_cards(state.hands[(state.turn + len(state.hands) - 1) % len(state.hands)], visible=False) # encode the person to the right (the previous player)
        ))

    def _encode_swap_state(self, state: State, drawn_card: Card) -> torch.Tensor:
        return torch.cat((
            self._encode_cards(state.hands[state.turn] + [drawn_card]),
            torch.tensor(probability.expected_values(state, see_hand=True), dtype=torch.float)
        ))
    
    # each cards is represented as a 40 bit value
    def _encode_cards(self, cards: list[Card], visible: bool = True) -> torch.Tensor:
        encoding = torch.zeros(len(cards) * 4, dtype=torch.float)
        
        for i, card in enumerate(cards):
            if visible or card.visible:
                encoding[i * 4 + card.suit.value] = card.val
        
        return encoding
    
    def take_turn(self, state: State) -> tuple[Action, int]:
        target_action: tuple[Action, int] = None

        state_tensor: torch.Tensor = self._encode_basic_state(state)
        swap_tensor: torch.Tensor | None = None

        if random.random() < self.epsilon:
            # don't save experience for random actions
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
        # Skip if no experiences to train on
        if not self.experiences:
            return
        
        # Calculate discounted rewards for each step
        # each past experience is less valuable than the more recent ones to the final reward
        gamma: float = 0.98  # discount factor
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
            if exp['swap_state'] is not None
        ]
        
        # expand swap loss scope for metrics
        swap_loss: torch.Tensor | None = None

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
        
        # I don't always want to save the metrics
        # self.metrics.add_metrics(action_loss, swap_loss, reward, len(self.experiences))
        
        # Clear experiences after training
        self.flush_experience()
    
    def plot_metrics(self):
        self.metrics.plot_metrics()
