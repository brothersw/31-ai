from agents.agent import Agent
from game.state import State
from game.action import Action

# Agent makes the decision to gain the most points from the next draw based on expected value
# It counts cards that are in the discard pile and that are already seen
class Greedy(Agent):
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        # just take a turn like normal, it will never need to handle under the gun in the first turn
        return self.take_turn(state)
    
    def take_turn(self, state: State) -> tuple[Action, int]:
        raise NotImplementedError()
    
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        # just take a turn like normal, it will never "call"
        return self.take_turn(state)