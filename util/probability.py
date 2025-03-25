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
    
    for card in [].join(state.hands):
        if card.suit == suit:
            if card.visible:
                count += card.val
                cards += 1

    total_count = 95 # 2 + 3 + ... + 10 + 10 + 10 + 10 + 11
    total_cards = 13 # 52 / 4

    return (total_count - count) / (total_cards - cards)

# returns a dictionary of the expected value for each suit
def expected_values(state: State) -> dict[Suit, float]:
    return {suit: expected_value(suit, state) for suit in Suit}