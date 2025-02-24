from game.card import Card
from game.action import Action

class Agent:
    def __init__(self):
        # hand gets filled by the game
        self.hand: list[Card] = []
        self.lives = 3
    
    def take_turn(self) -> tuple[Action, int]:
        raise NotImplementedError()
    
    # include additional under the gun logic
    def take_first_turn(self) -> tuple[Action, int]:
        raise NotImplementedError()
    
    # remove calling logic
    def take_last_turn(self) -> tuple[Action, int]:
        raise NotImplementedError()
