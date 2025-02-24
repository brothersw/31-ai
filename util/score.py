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
