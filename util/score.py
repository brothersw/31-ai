from agents.agent import Agent
from game.card import *

def score(hand: list[Card]) -> int:
    assert(len(hand) == 3)
    
    score = 0

    for e in Suit:
        s = 0
        for c in hand:
            if e == c.suit:
                s += c.val
        score = max(s, score)

    return score

# returns a list of rewards for each player
def get_rewards(players: list[tuple[Agent, int]], caller: int) -> list[float]:
    # Find the lowest score and its index
    scores = [score for _, score in players]
    min_score = min(scores)
    max_score = max(scores)
    
    rewards = []
    for i, (agent, player_score) in enumerate(players):
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
        
        rewards.append(reward)
    
    return rewards
