from agents.agent import Agent
from game.action import Action

class Human(Agent):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def take_turn(self) -> tuple[Action, int]:
        raise NotImplementedError()
    
    def take_first_turn(self) -> tuple[Action, int]:
        return self.take_turn()
    
    # remove calling logic
    def take_last_turn(self) -> tuple[Action, int]:
        raise NotImplementedError()
