# Table of Contents


- [Overview](#overview)
	- [Running the code](#running%20the%20code)
- [Gameplay](#gameplay)
  - [Game State](#game%20state)
  - [Agent Interfaces](#agent%20interfaces)
    - [Adversarial Strategies](#adversarial%20strategies)
- [The Model](#the%20model)
  - [Architecture](#architecture)
    - [Input Representation](#input%20representation)
    - [Multi-Agent Considerations](#multi-agent%20considerations)
    - [Action Space](#action%20space)
      - [Masking](#masking)
  - [Training](#training)
    - [Epsilon](#epsilon)
    - [Reward Function](#reward%20function)
    - [Experiences](#experiences)
    - [Reward Discounts](#reward%20discounts)
    - [Training Procedure](#training%20procedure)
    - [Computational Requirements](#computational%20requirements)
  - [Metrics](#metrics)
    - [Wins](#wins)
    - [Granular Metrics](#granular%20metrics)
  - [Saving and Loading](#saving%20and%20loading)
- [Evaluating Success](#evaluating%20success)
- [Divergence From Proposal](#divergence%20from%20proposal)
- [Improvement](#improvement)
- [Sources](#sources)

# Overview

The goal of this project was to create a neural network to play the card game 31.

# Gameplay

![[InitialGameplay.png]]

31 is a card game where each player starts with three cards. Each player starts with 3 lives. Players take turns drawing from either the deck or discard pile, then swapping one of their cards with the drawn card. The objective is to have the highest sum of card values in any single suit. A player may "call" to end the round when confident in their hand's strength, forcing all players to reveal their hands after all other players have one more turn. The player with the lowest score loses a life, and the game continues until only one player remains.

Additional rules include: 

- "under the gun", where a player may "call" on the very first turn of the game. If this happens, the round immediately ends in players revealing their hands and lives being lost.
- if a player that scores the maximum hand of 31 points, all other players lose a live
- if a player "calls" and they are the person who lost, they lose 2 lives instead of 1

![[TakeTurn.png]]

#### Running the code

This code was developed on python 3.11

Optionally create a python virtual environment to avoid polluting a global python environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install packages:
```bash
pip -r requirements.txt
```

Run the code:
```bash
python main.py
```

## Game State

A round's state is represented as follows:
- A shuffled deck of 52 regular playing cards
- Each player's hand
- A discard pile (face up)
- The current turn indicator
- A first turn flag
- If the game has been "called"

```python
class State:
    def __init__(self, num_players: int):
        self.deck: list[Card] = self._create_deck()
        self.hands: list[list[Card]] = self._build_hands(num_players)
        self.discard: list[Card] = [self.deck.pop()]
        self.discard[-1].visible = True

        self.turn: int = 0
        self.first_turn: bool = True
        self.called: int = -1
```

> When the game gets "called", the called variable changes to represent the turn of the player who "called"

Note that cards hold the visible attribute which tracks if the card has been seen before. This is useful for the model being able to simulate counting cards if they have been seen face-up in the discard pile. This is a highly viable strategy for humans because it usually just means tracking the 3 cards in the player to the left and right of the player.

## Agent Interfaces

I have called all simulated players agents. All agents implement the following interface in order to work with the game. This allows all agents to:

- Take normal turns
- Choose how to handle the first and last turns of the game differently

The game expects tuples of an action paired with what to do with the cards in each action. e.g. (Draw, 1) would replace the card in the player's hand of index 1 with the top of the draw pile.

```python
class Agent:
    def __init__(self):
        # hand gets filled by the game state
        self.lives: int = 3
    
    def take_turn(self, state) -> tuple[Action, int]:
        raise NotImplementedError()
    
    # include additional under the gun logic
    def take_first_turn(self, state) -> tuple[Action, int]:
        raise NotImplementedError()
    
    # remove calling logic
    def take_last_turn(self, state) -> tuple[Action, int]:
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        return f"{type(self)} [{self.lives}]"
```

Additionally, the agent has the function `revive` that is already implemented and the function `train` that is optional to implement.

### Adversarial Strategies

The model was trained against several baseline strategies including random play and greedy algorithms that always choose the highest-value immediate action. This adversarial training helps the model develop strategies against opponents. Originally, I began training against a random agent until the model worked adn then the agent that that was most valuable to train against was the greedy agent. Additionally, I tried to create a greedy agent that knows when to "call"; however, it only under-performed the greedy agent in both capabilities and training performance.

# The Model

Testing and training was conducted through hundreds of thousands of simulated rounds broken up into batches. Each batch represents 1000 rounds of play. In total, I have run several million rounds of play when testing these models.

## Architecture

The model consists of 2 neural networks, the ActionNN and the SwapNN. The action model decides whether to draw from the deck, the discard, or to call, while the swap model decides which cards to keep and which cards to discard. The models are simple feed-forward models that are 5 and 4 layers deep respectively, with hidden sizes of 256 and 128 neurons respectively. These sizes for the model were chosen after expanding the model size until diminishing returns were found.

```python
# neural network to select the action to take
class ActionNN(nn.Module):
    def __init__(self, input_size: int = 4 * 4 + 4 + 2 + 4 * 6, hidden_size: int = 256):
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
```

### Input Representation

Cards are represented as an 4 item tensor where each item corresponds to a suit and the value represents the card's value.

For example, a 2 of spades would be represented as `[2, 0, 0, 0]` and a 8 of clubs would be `[0, 8, 0, 0]`

```python
def _encode_cards(self, cards: list[Card], visible: bool = True) -> torch.Tensor:
    encoding = torch.zeros(len(cards) * 4, dtype=torch.float)

    for i, card in enumerate(cards):
        if visible or card.visible:
            encoding[i * 4 + card.suit.value] = card.val

    return encoding
```

The action model takes in a tensor of floats representing the following things:
- The player's hand
- The top of the discard pile
- The expected values from each suit from the draw pile
- If it is the first turn of the game
- If the game has been "called"
- The cards of the player to the left and right of the agent if the card has been seem before
  (what would happen with a player that counts cards another picks up from the discard)

The swap model takes in a tensor of floats representing the following things:
- The player's hand
- The expected values of each card that could be drawn
- The player to the left's hand (the downstream player) if the card has been seem before

#### Multi-Agent Considerations

Notice that the only players that are directly encoded are the players to the left and to the right of the agent. This is because those are the only two players that can directly influence the agent through handing cards in the discard. Thus, strategy is best spent on those two players. The `probability.expected_values` function is able to count seen cards for the rest of the players in the game.

### Action Space

The output action space of the models is as follows:

The action model outputs a tensor of 3 floats representing confidence in the possible actions:

```
[DRAW, DRAW_DISCARD, CALL]
```

The swap model outputs a tensor of 4 floats representing confidence in the which cards to swap or whether or not to swap at all:

```
[1st card, 2nd card, 3rd card, don't swap]
```

#### Masking

Illegal actions are masked out of the action space by making the value of that output tensor negative infinity. For example, a player cannot call if the game is in the last round due to another calling. Another example might be a player not able to pick up the top card from the visible discard pile only to place it back down again on the top of the discard pile.

```python
if state.called != -1:
    action_values[Action.CALL.value] = float('-inf')
```

## Training

Training data is gathered by playing the model against the other programmed models.

Self-play training (pitting the model against itself) such as what AlphaZero does did not generate a significantly better model while dramatically increasing training time and compute needed. Ultimately, similar results were found when doing final bench marking against the other strategies.

(suragnair, 2017)

### Epsilon

An epsilon representing the chance to choose a random action is needed during training. This is to keep the model from reaching a local minima in its strategy, leaving it the ability to randomly jump out of that local minima. An epsilon of zero leads to premature flatlines in capability. An epsilon that is too small leads to premature flatlines that eventually jump out of the minima it is in hindering training time by wasting time in the minima. An epsilon that is too big renders the model too erratic to perform at its best.

Overall, I found that an epsilon of 4-5% lead to the best results in training and then reducing that epsilon to 0% after training lead to the best ability to benchmark and play against the model.

(Dabney, 2020)

### Reward Function

The reward function is designed to reward winning the game, increasing in amount of reward the greater the margin of victory. If the agent successfully got the maximum score or was the "caller", greater rewards are applied.

It punishes losing the game, incurring greater punishments for losing multiple lives or going out of the game entirely.

```python
reward = 0.0

reward += (player_score - min_score) * 0.5  # performance relative to min_score

if player_score == min_score:
    if i == caller:
        # Heavily penalize calling and losing
        reward -= 5.0
    else:
        # Regular penalty for losing
        reward -= 2.0
elif player_score == 31:
    # Significant reward for achieving perfect score
    reward += 10.0
elif i == caller and player_score > min_score:
    # Reward for successful call based on margin of victory
    reward += (player_score - min_score) * 0.3

# Progressive penalties based on lives lost
lives_penalty = (3 - agent.lives) * -0.3
reward += lives_penalty

# Game-ending states
if agent.lives <= 0:
    reward -= 2.0  # penalty for being eliminated

# Small bonus for staying alive
if agent.lives > 0:
    reward += 0.5
```

### Experiences

Since training has to be done after the round is done, each action is accumulated as an "experience" to be trained on later. Experiences passed into training the model based on the actions taken.

```python        
# Filter only for experiences that involved drawing
swap_experiences: list[tuple[torch.Tensor, int, float]] = [
    (exp['swap_state'], exp['position'], reward)
    for exp, reward in zip(self.experiences, discounted_rewards)
    if exp['swap_state'] is not None
]
```

### Reward Discounts

All rewards are "discounted" over time. This is because training is done after several actions have already taken place. I only have one reward to apply to multiple actions and the actions that occur at the end of a match are much more likely to be decisive towards the outcome at the end. Therefore, actions closest to the end were worth the most, each subsequent action decreasing in value by being multiplied against the discount rate. A discount factor of 98% is what I settled on after some trials with various discount factors.

$$d(n) = r \cdot 0.98^n$$

```python
# Calculate discounted rewards for each step
# each past experience is less valuable than the more recent ones to the final reward
gamma: float = 0.98  # discount factor
discounted_rewards: list[float] = []
running_reward: float = reward
for _ in range(len(self.experiences)):
    discounted_rewards.insert(0, running_reward)
    running_reward *= gamma
```

The relative worth of actions with a discount factor of 98% is:

![[DiscountedRewards.png]]

(Mahadevan, 1996)

### Training Procedure

After experiences are collected, each action is assigned its rewards based on the discounted rewards. The tensor representing the relation from the input to the output is then passed into a loss function (mean squared error) along with the rewards to generate the direction the model should improve in through PyTorch's optimizer working to minimize the mean squared error.

```python
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
```

(Bartlett, 1998)

### Computational Requirements

The model can be run purely CPU-side on a few megabytes of RAM. Training was multi-threaded through PyTorch and can be done on the GPU with a few tweaks; however, my laptop's integrated GPU was not showing greater performance in training so everything is handed CPU-side. This should have increased training time; however, with the relatively small model that I am training it was not an issue.

## Metrics

These metrics helped me benchmark changes I made to the model. All improvements to the model had to correspond to an improvement to one of the metrics that I was watching.

### Wins

The most important metric to me was how well the model did compared to the other strategies that I had programmed. When the model begins to flatline, that is the point at which I stopped training and used the nearest checkpoint to play against. The reason I didn't let it train further past the beginning of the flatline is that the model would begin to overfit the strategies present in training at that point and would have decreased performance against any other strategies that it had not trained against.

![[Wins.png]]

### Granular Metrics

As the models train, I was logging data to debug and hunt for improvements in the model architecture. The custom logger I added captures the following:

```python
class TrainingMetrics:
    def __init__(self):
        self.action_losses: list[float] = []
        self.swap_losses: list[float] = []
        self.rewards: list[float] = []
        self.episode_lengths: list[int] = []
```

I can graph the rolling average in a specified window or individual values for each of these metrics which get generated on each training iteration.

This was useful in finding patterns in training data. For example, I used to base reward metrics off of a normalized scale compared to the worst/best player; however, when training in matches with 2 players that lead to very little variation in the reward function's output aside from the difference between a win and a loss.

![[Metrics.png]]

## Saving and Loading

The models, optimizes, and epsilon are saved via torch's built-in save functionality. 
```python
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
```

# Evaluating Success

The wins metric is a great indicator of overall success of the model and is as good of an objective benchmark as I can collect. On average, the model wins **74.6 percent** of the time compared to my best programmed strategy (the greedy strategy) in 1v1 matches.

> Note the greater ability for the neural network to win if it is only pitted against the greedy strategy without a third random player like present in the wins benchmark earlier discussed. This is because the random agent will detonate rounds prematurely leading to a smaller skill difference visible in that graph.

A less objective but more real-world benchmark is playing against real players. In these trials, the model won 65% of matches against me in 40 trials.

Overall, I would consider this model an outstanding success in exhibiting the ability to learn to play this card game.

# Divergence From Proposal

Originally, I had proposed an ELO rating system to compare models; however, an ELO rating system proved more complex than needed for model evaluation.

Furthermore, a mixture-of-experts style model is not a truly apt characterization of the model architecture since the action model has both decision making capabilities and chooses to use the swap model as a consequence of its decisions.

# Improvement

The model struggles most with the secondary considerations that the swap model could take. For example, drawing a 6 of spades with a hand containing a Queen of spades, a 4 of clubs, and a 2 of clubs has a greater probability than I would like to replace the 4 of clubs with a 6 of spades instead of a marginally better strategic choice of replacing the 2 of clubs. This is an improvement that a human would immediately be able to recognize; however, the lack of secondary scoring objectives and minimal benefit limits the swap model to making a sub-optimal choice.

Additionally, an expanded model architecture with finer-tuned reward strategies that individually target each neural network could potentially provide training improvements.

# Sources

Bartlett, P. (1998). The sample complexity of pattern classification with neural networks: the size of the weights is more important than the size of the network. IEEE Transactions on Information Theory, 44(2), 525-536. 

Dabney, W., Ostrovski, G., & Barreto, A. (2020). Temporally-Extended ε-Greedy Exploration. ArXiv (Cornell University). https://doi.org/10.48550/arxiv.2006.01782

Mahadevan, S. (1996). Average reward reinforcement learning: Foundations, algorithms, and empirical results. _Machine Learning_, _22_, 159–195. https://doi.org/10.1023/a:1018064306595

PyTorch. (n.d.). PyTorch documentation — PyTorch master documentation. Pytorch.org. https://pytorch.org/docs/stable/index.html

suragnair. (2017). GitHub - suragnair/alpha-zero-general: A clean implementation based on AlphaZero for any game in any framework + tutorial + Othello/Gobang/TicTacToe/Connect4 and more. GitHub. https://github.com/suragnair/alpha-zero-general
