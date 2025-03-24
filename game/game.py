from agents.agent import Agent
from game.action import Action
from game.state import State
from util import score

# TODO: unhiding cards is broken

class Game:
    def __init__(self, players: list[Agent]):
        self.state: State = State(len(players))
        self.players = players
    
    # the main game runner that picks the turn for the agent to take
    # if the game is ended, returns true
    def pick_turn(self) -> bool:
        agent = self.players[self.state.turn]

        print(f"The current turn is: {self.state.turn}")
        
        # under the gun logic
        if self.state.first_turn:
            action = agent.take_first_turn(self.state)
            self.state.first_turn = False
            if action[0] == Action.CALL:
                self.state.called = self.state.turn
                return True
        # normal turn
        elif self.state.called == -1:
            action = agent.take_turn(self.state)
        # last turn
        else:
            action = agent.take_last_turn(self.state)
            assert action[0] != Action.CALL
        
        self._take_action(action)
        self._next_turn()

        # if the game is ending normally
        return self.state.called == self.state.turn

    # takes in a tuple containing the actiont to take and what card to replace
    # in the case of a call, the second int's value doesn't matter
    def _take_action(self, action: tuple[Action, int]):
        match action[0]:
            case Action.CALL:
                self.state.called = self.state.turn
            case Action.DRAW:
                assert len(self.state.deck) >= 1
                assert action[1] >= 0 and action[1] <= 3
                
                # discard the just drawn card
                if action[1] == 3:
                    self.state.discard.append(self.state.deck.pop())
                else:
                    self.state.discard.append(self.state.hands[self.state.turn][action[1]])
                    self.state.hands[self.state.turn][action[1]] = self.state.deck.pop()
                    self.state.discard[-1].visible = True

            case Action.DRAW_DISCARD:
                assert len(self.state.discard) >= 1
                assert action[1] >= 0 and action[1] <= 2

                discarded = self.state.hands[self.state.turn][action[1]]
                self.state.hands[self.state.turn][action[1]] = self.state.discard.pop()
                self.state.discard.append(discarded)
                self.state.discard[-1].visible = True
            case _:
                raise ValueError("Unknown action taken")

    # increments and wraps the turn counter
    def _next_turn(self):
        assert self.state.turn >= 0 and self.state.turn < len(self.players)
        
        self.state.turn += 1
        self.state.turn %= len(self.players)

    # ends the game for a normal ending
    # the person with the least score loses a life
    # if the person who called is least, two lives are lost
    def end_game(self):
        assert self.state.called != -1
        
        losers: list[tuple[Agent, int]] = self._find_losers()
        print(f"The losers are {losers}")
        for loser in losers:
            if loser[0] == self.players[self.state.called]:
                loser[0].lives -= 2
            else:
                loser[0].lives -= 1

            if loser[0].lives <= 0:
                loser[0].lives = 0

    # returns a list (handling ties) contining tuples of (Agent, score) of those who lost
    def _find_losers(self) -> list[tuple[Agent, int]]:
        worst: list[tuple[Agent, int]] = [(None, 100)]
        for i, p in enumerate(self.players):
            s = score.score(self.state.hands[i])
            if s == 31:
                return self._find_losers_31()
            elif s < worst[0][1]:
                worst = [(p, s)]
            elif s == worst[0][1]:
                worst.append((p, s))

        return worst

    # returns all players who don't have a score of 31
    def _find_losers_31(self) -> list[tuple[Agent, int]]:
        raise NotImplementedError()