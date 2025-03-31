from game.card import Suit
from game.state import State

# TODO: test this
# returns the expected value to get this suit from the draw pile
def expected_value(suit: Suit, state: State) -> float:
    # count the number of cards in the deck that are this suit
    count = 0
    cards = 0

    for card in state.discard:
        if card.suit == suit:
            count += card.val
            cards += 1
    
    for hand in state.hands:
        for card in hand:
            if card.suit == suit:
                if card.visible:
                    count += card.val
                    cards += 1
    
    total_count = 95 # 2 + 3 + ... + 10 + 10 + 10 + 10 + 11
    total_cards = 13 # 52 / 4

    return (total_count - count) / (total_cards - cards)

# returns a list of the expected value for each suit in the order of the Suit enum
def expected_values(state: State) -> list[float]:
    return [expected_value(suit, state) for suit in Suit]