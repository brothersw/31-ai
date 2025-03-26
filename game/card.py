from enum import Enum

class Suit(Enum):
    SPADE = 0
    CLUB = 1
    DIAMOND = 2
    HEART = 3

class Card:
    def __init__(self, val: int, suit: Suit):
        self.val: int = val
        self.suit: Suit = suit
        self.visible: bool = False
    
    def get_neural_repr(self) -> tuple[int, int]:
        if self.visible:
            return (self.suit.value, self.val)
        else:
            return (0, 0)

    def __repr__(self) -> str:
        char = ""
        match self.suit:
            case Suit.SPADE:
                char = "S"
            case Suit.CLUB:
                char = "C"
            case Suit.DIAMOND:
                char = "D"
            case Suit.HEART:
                char = "H"

        if self.visible:
            return f" {char}:{self.val} "
        else:
            return f"#{char}:{self.val}#"
