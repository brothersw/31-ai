import unittest
from game.game import Game
from game.state import State
from game.action import Action
from agents.agent import Agent

# python -m unittest tests.test_logic

class MockAgent(Agent):
    def __init__(self):
        super().__init__()
        self.lives = 3
        
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        return (Action.DRAW, 0)
        
    def take_turn(self, state: State) -> tuple[Action, int]:
        return (Action.DRAW, 0)
        
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        return (Action.DRAW, 0)

class TestGameLogic(unittest.TestCase):
    def setUp(self):
        self.players = [MockAgent() for _ in range(3)]
        self.game = Game(self.players)

    def test_init_state(self):
        self.assertEqual(len(self.game.state.deck), 52 - 10)  # 52 - (3 players * 3 cards + 1 discard)
        self.assertEqual(len(self.game.state.hands), 3)
        for hand in self.game.state.hands:
            self.assertEqual(len(hand), 3)
        self.assertEqual(len(self.game.state.discard), 1)
        self.assertEqual(self.game.state.turn, 0)
        self.assertTrue(self.game.state.first_turn)
        self.assertEqual(self.game.state.called, -1)

    def test_turn_progression(self):
        self.game._next_turn()
        self.assertEqual(self.game.state.turn, 1)
        self.game._next_turn()
        self.assertEqual(self.game.state.turn, 2)
        self.game._next_turn()
        self.assertEqual(self.game.state.turn, 0)
        self.game._next_turn()
        self.assertEqual(self.game.state.turn, 1)

    def test_draw_action(self):
        initial_deck_size = len(self.game.state.deck)
        initial_discard_size = len(self.game.state.discard)
        initial_card = self.game.state.hands[0][0]
        
        self.game._take_action((Action.DRAW, 0))
        
        self.assertEqual(len(self.game.state.deck), initial_deck_size - 1)
        self.assertEqual(len(self.game.state.discard), initial_discard_size + 1)
        self.assertNotEqual(self.game.state.hands[0][0], initial_card)
        self.assertTrue(self.game.state.discard[-1].visible)

    def test_draw_discard_action(self):
        top_discard = self.game.state.discard[-1]
        initial_hand_card = self.game.state.hands[0][0]
        initial_discard_size = len(self.game.state.discard)
        
        self.game._take_action((Action.DRAW_DISCARD, 0))
        
        self.assertEqual(self.game.state.hands[0][0], top_discard)
        self.assertEqual(self.game.state.discard[-1], initial_hand_card)
        self.assertEqual(len(self.game.state.discard), initial_discard_size)
        self.assertTrue(self.game.state.discard[-1].visible)

    def test_call_action(self):
        self.game._take_action((Action.CALL, 0))
        self.assertEqual(self.game.state.called, self.game.state.turn)