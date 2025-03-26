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
    
    rewards = []
    for i, (agent, player_score) in enumerate(players):
        # Base reward is the difference between this player's score and the lowest score
        reward = (player_score - min_score) * 0.1
        
        # If this player has the lowest score
        if player_score == min_score:
            # Double negative reward if they were the caller
            if i == caller:
                reward = -2
            else:
                reward = -1
        # increase reward if the player got 31 (making all other players lose a life)
        elif player_score == 31:
            reward += 5
        # Add bonus reward for the caller if they weren't the loser
        elif player_score > min_score and i == caller:
            reward += 1.0
        
        # exaspertabe negative reward if the player is out of the game
        if agent.lives <= 0:
            reward += -5
            
        rewards.append(reward)
    
    return rewards