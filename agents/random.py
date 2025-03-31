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
        return self._random_action(state)

    def take_last_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)

    def _random_action(self, state: State) -> tuple[Action, int]:
        if state.called != -1:
            action = random.choice([Action.DRAW, Action.DRAW_DISCARD])
        else:
            action = random.choice([Action.DRAW, Action.DRAW_DISCARD, Action.CALL])

        idx = 0        
        if action == Action.DRAW:
            idx = random.randint(0, 3)
        elif action == Action.DRAW_DISCARD:
            idx = random.randint(0, 2)
        
        return (action, idx)