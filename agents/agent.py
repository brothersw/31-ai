from game.action import Action

class Agent:
    def __init__(self):
        # hand gets filled by the game state
        self.lives: int = 3
    
    def take_turn(self, state) -> tuple[Action, int]:
        raise NotImplementedError()
    
    # include additional under the gun logic
    def take_first_turn(self, state) -> tuple[Action, int]:
        raise NotImplementedError()
    
    # remove calling logic
    def take_last_turn(self, state) -> tuple[Action, int]:
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        return f"{type(self)} [{self.lives}]"