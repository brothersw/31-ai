from game.card import Suit
from game.state import State

# returns the expected value to get this suit from the draw pile
def expected_value(suit: Suit, state: State, seeHand: bool) -> float:
    # count the number of seen_cards in the deck that are this suit
    count = 0
    seen_cards = 0
    total_count = 95 # 2 + 3 + ... + 10 + 10 + 10 + 10 + 11
    total_cards = 52

    for card in state.discard:
        seen_cards += 1
        if card.suit == suit:
            count += card.val
    
    for i, hand in enumerate(state.hands):
        for card in hand:
            if card.visible or (seeHand and i == state.turn):
                seen_cards += 1
                if card.suit == suit:
                    count += card.val
    
    if seen_cards == 52:
        return 0

    return (total_count - count) / (total_cards - seen_cards)

# returns a list of the expected value for each suit in the order of the Suit enum
def expected_values(state: State, see_hand: bool = False) -> list[float]:
    return [expected_value(suit, state, see_hand) for suit in Suit]
