import random
from agents.agent import Agent
from game.state import State
from game.action import Action

class Random(Agent):
    def __init__(self):
        super().__init__()
    
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)

    def take_turn(self, state: State) -> tuple[Action, int]:
        action: Action = random.choice([Action.DRAW, Action.DRAW_DISCARD, Action.CALL])
        if action == Action.CALL:
            return (action, 0)
        else:
            return (action, random.randint(0, 2))
    
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        action: Action = random.choice([Action.DRAW, Action.DRAW_DISCARD])
        return (action, random.randint(0, 2))
