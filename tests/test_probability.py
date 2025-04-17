import unittest
from game.state import State
import util.probability as probability

class TestProbability(unittest.TestCase):
    def setUp(self):
        self.state = State(2)

    def test_blank_state(self):
        # I don't want any globablly known cards
        self.state.deck.append(self.state.discard.pop())

        probs = probability.expected_values(self.state)
        
        for p in probs:
            self.assertAlmostEqual(p, 95 / 52)

    def test_hand_state(self):
        # I don't want any globablly known cards
        self.state.deck.append(self.state.discard.pop())


        probs: list[float] = probability.expected_values(self.state, see_hand=True)

        for i, p in enumerate(probs):
            card_vals: int = 0
            for card in self.state.hands[self.state.turn]:
                if card.suit.value == i:
                    card_vals += card.val
            num_unknown_cards = 52 - len(self.state.hands[self.state.turn])
            self.assertAlmostEqual(p, (95 - card_vals) / num_unknown_cards)
