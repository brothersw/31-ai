import unittest
import torch
from agents.ai import AIAgent
from game.card import Card, Suit
from game.state import State

# python -m unittest tests.test_encoding

class TestEncoding(unittest.TestCase):
    def test_encoding_shape(self):
        ai: AIAgent = AIAgent()
        state: State = State(1)
        encoded: torch.Tensor = ai._encode_basic_state(state)
        self.assertEqual(encoded.shape, (40 * 4 + 2,))

    def test_encoding_values(self):
        ai: AIAgent = AIAgent()
        state: State = State(1)
        state.turn = 0
        state.hands = [[Card(2, Suit.SPADE), Card(3, Suit.SPADE), Card(4, Suit.SPADE)]]
        encoded: torch.Tensor = ai._encode_swap_state(state, Card(9, Suit.CLUB))
        self.assertEqual(encoded.shape, (40 * 4,))
        self.assertEqual(encoded[0], 1) # 2 of spades in first slot
        self.assertEqual(encoded[1], 0)
        self.assertEqual(encoded[40 * 1], 0)
        self.assertEqual(encoded[40 * 1 + 1], 1) # 3 of spades in second slot
        self.assertEqual(encoded[40 * 1 + 2], 0)
        self.assertEqual(encoded[40 * 2 + 2], 1) # 4 of spades in the third slot
        self.assertEqual(encoded[40 * 3 + 10 * 1 + 6], 0)
        self.assertEqual(encoded[40 * 3 + 10 * 1 + 7], 1) # 9 of clubs in the fourth slot

    def test_encoding_values_2(self):
        ai: AIAgent = AIAgent()
        state: State = State(1)
        state.turn = 0
        state.hands = [[Card(10, Suit.SPADE), Card(10, Suit.SPADE), Card(10, Suit.SPADE)]]
        encoded: torch.Tensor = ai._encode_swap_state(state, Card(10, Suit.HEART))
        self.assertEqual(encoded.shape, (40 * 4,))
        self.assertEqual(encoded[40 * 0 + 8], 1) # 10 of spades in first slot
        self.assertEqual(encoded[40 * 0 + 9], 0)
        self.assertEqual(encoded[40 * 1 + 8], 1) # 10 of spades in second slot
        self.assertEqual(encoded[40 * 1 + 9], 0)
        self.assertEqual(encoded[40 * 2 + 8], 1) # 10 of spades in third slot
        self.assertEqual(encoded[40 * 2 + 9], 0)
        self.assertEqual(encoded[40 * 3 + 10 * 3 + 8], 1) # 10 of hearts in fourth slot
        self.assertEqual(encoded[40 * 3 + 10 * 3 + 9], 0)