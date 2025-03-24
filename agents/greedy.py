from agents.agent import Agent
from game.state import State
from game.action import Action

class Greedy(Agent):
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        raise NotImplementedError()
    
    def take_turn(self, state: State) -> tuple[Action, int]:
        raise NotImplementedError()
    
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        raise NotImplementedError()