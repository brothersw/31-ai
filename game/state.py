from game.card import Card
from game.card import Suit
import random

class State:
    def __init__(self, num_players: int):
        self.deck: list[Card] = self._create_deck()
        self.hands: list[list[Card]] = self._build_hands(num_players)
        self.discard: list[Card] = [self.deck.pop()]
        self.discard[-1].visible = True

        self.turn: int = 0
        self.first_turn: bool = True
        self.called: int = -1


    # puts 3 cards from the deck in each person's hands
    def _build_hands(self, num_players: int) -> list[list[Card]]:
        assert num_players < len(self.deck) // 3

        return [[self.deck.pop() for _ in range(0, 3)] for _ in range(0, num_players)]

    # builds 52 card deck
    def _create_deck(self) -> list[Card]:
        deck = []

        for e in Suit:
            # number cards
            for i in range(2, 10):
                deck.append(Card(i, e))
            
            # face cards
            for _ in range(0, 4):
                deck.append(Card(10, e))
            # ace
            deck.append(Card(11, e))

        random.shuffle(deck)
        return deck
    
    def __repr__(self) -> str:
        return f"Turn: {self.turn}, Called: {self.called}\n Discard: {self.discard[-1]}, Hands: {self.hands}"