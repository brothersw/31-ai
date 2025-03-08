from agents.agent import Agent
from game.action import Action
from game.state import State

class Human(Agent):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def take_turn(self, state: State) -> tuple[Action, int]:
        print(f" Your hand is: {state.hands[state.turn]}")
        print("What do you want to do?")
        print("[0] Draw top card")
        print(f"[1] Draw top discard: {state.discard[-1]}")
        print("[2] Call")

        act_n: int = int(input("> "))
        # pick again on bad input
        if act_n > 2 or act_n < 0:
            print("BAD INPUT")
            return self.take_turn(state)
        
        act: Action = _get_action(act_n)
        
        print("")

        match act:
            case Action.DRAW:
                print(f"You drew: {state.deck[-1]}")
                return (act, self._replace_dialog_deck(state))
            case Action.DRAW_DISCARD:
                return (act, self._replace_dialog_discard(state))
            case Action.CALL:
                return (act, 0)
            case _:
                raise ValueError()
    
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        return self.take_turn(state)
    
    # remove calling logic
    def take_last_turn(self, state: State) -> tuple[Action, int]:
        print("What do you want to do?")
        print("[0] Draw top card")
        print(f"[1] Draw top discard: {state.discard[-1]}")
        
        act_n = int(input("> "))
        # pick again on bad input
        if act_n > 1 or act_n < 0:
            print("BAD INPUT")
            return self.take_last_turn(state)
        
        act = _get_action(act_n)
        
        match act:
            case Action.DRAW:
                print(f"You drew: {state.deck[-1]}")
                return (act, self._replace_dialog_deck(state))
            case Action.DRAW_DISCARD:
                return (act, self._replace_dialog_discard(state))
            case _:
                raise ValueError()
    
    def _replace_dialog_deck(self, state: State) -> int:
        print("What card do you want to replace?")
        print(f"[0] {state.hands[state.turn][0]}")
        print(f"[1] {state.hands[state.turn][1]}")
        print(f"[2] {state.hands[state.turn][2]}")
        print(f"[3] Discard the just drawn card")

        ret = int(input("> "))
        # pick again on bad input
        if ret > 3 or ret < 0:
            print("BAD INPUT")
            return self._replace_dialog_deck(state)

        return ret

    def _replace_dialog_discard(self, state: State) -> int:
        print("What card do you want to replace?")
        print(f"[0] {state.hands[state.turn][0]}")
        print(f"[1] {state.hands[state.turn][1]}")
        print(f"[2] {state.hands[state.turn][2]}")

        ret = int(input("> "))
        # pick again on bad input
        if ret > 2 or ret < 0:
            print("BAD INPUT")
            return self._replace_dialog_discard(state)

        return ret

    def __repr__(self) -> str:
        return f"{self.name} [{self.lives}] holds: {state.hands[state.turn]}"

def _get_action(num: int) -> Action:
    match num:
        case 0:
            return Action.DRAW
        case 1:
            return Action.DRAW_DISCARD
        case 2:
            return Action.CALL
        case _:
            raise ValueError()
