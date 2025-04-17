from agents.agent import Agent
from game.state import State
from game.action import Action
from game.card import Card
from game.card import Suit
import util.probability as probability
import util.score as score

# Agent makes the decision to gain the most points from the next draw based on expected value
# It counts cards that are in the discard pile and that are already seen
# It will call if it sees that it can't get a better value on average in the next turn than its current hand
class Optimizer(Agent):
    def take_first_turn(self, state: State) -> tuple[Action, int]:
        # just take a turn like normal, it will never need to handle under the gun in the first turn
        return self.take_turn(state)
    
    def take_turn(self, state: State) -> tuple[Action, int]:
        hand = state.hands[state.turn]
        if score.score(hand) == 31 and state.called == -1:
            return (Action.CALL, 0)

        discard_card = state.discard[-1]
        best_action = Action.CALL if state.called == -1 else Action.DRAW
        best_idx = 0
        
        test_idx, best_score = self.swap_card(hand, discard_card)
        
        # if swapping from the discard pile is an improvement
        if test_idx != 3:
            best_idx = test_idx 
            best_action = Action.DRAW

        draw_probabilities = probability.expected_values(state, see_hand=True)
        for i, suit in enumerate(Suit):
            test_card = Card(suit, draw_probabilities[i])
            test_idx, test_score = self.swap_card(hand, test_card)
            
            # if swapping from the discard pile is an improvement
            if test_idx != 3 and test_score > best_score:
                best_score = test_score
                best_action = Action.DRAW
        
        # if I am actually drawing from the discard pile, pick what gets swapped
        if best_action == Action.DRAW:
            best_idx, _ = self.swap_card(state.hands[state.turn], state.deck[-1])

        return (best_action, best_idx)
    
    # returns card to replace, score when replacing
    def swap_card(self, hand: list[Card], new_card: Card):
        best_swap: int = 3
        max_score: int = score.score(hand)
        
        # Test swapping each card in the hand
        for i in range(len(hand)):
            # Create a temporary hand with the card swapped
            temp_hand = hand.copy()
            temp_hand[i] = new_card

            # Calculate the score of the temporary hand
            current_score = score.score(temp_hand)

            # Update best swap if this score is better
            if current_score > max_score:
                max_score = current_score
                best_swap = i

        return best_swap, max_score


    def take_last_turn(self, state: State) -> tuple[Action, int]:
        # just take a turn like normal, it will never "call"
        return self.take_turn(state)
