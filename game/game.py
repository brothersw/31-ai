import random
from agents.agent import Agent
from game.action import Action
from game.state import State
from util import score

class Game:
    def __init__(self, players: list[Agent]):
        self.state: State = State(len(players))
        self.players = players
    
    # the main game runner that picks the turn for the agent to take
    # if the game is ended, returns true
    def pick_turn(self) -> bool:
        agent = self.players[self.state.turn]

        print(self.state)
        
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

                # if the deck was emptied, shuffle the discard pile and reset it
                if len(self.state.deck) == 0:
                    self.state.deck = self.state.discard
                    random.shuffle(self.state.deck)
                    self.state.discard = [self.state.deck.pop()]

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
    # apply rewards to each player
    def end_game(self):
        assert self.state.called != -1

        scores: list[tuple[Agent, int]] = [(p, score.score(self.state.hands[i])) for i, p in enumerate(self.players)]
        
        losers: list[tuple[Agent, int]] = self._find_losers(scores)
        # print(f"The losers are {losers}")
        for loser in losers:
            if loser[0] == self.players[self.state.called]:
                loser[0].lives -= 2
            else:
                loser[0].lives -= 1

            if loser[0].lives <= 0:
                loser[0].lives = 0
        
        # If all players have 0 or fewer lives, give everyone a life (up to max 3)
        if all(p.lives <= 0 for p in self.players):
            for p in self.players:
                p.lives = min(p.lives + 1, 3)

        rewards = score.get_rewards(scores, self.state.called)
        for i, reward in enumerate(rewards):
            self.players[i].train(reward)
        
    # returns a list (handling ties) contining tuples of (Agent, score) of those who lost
    def _find_losers(self, scores: list[tuple[Agent, int]]) -> list[tuple[Agent, int]]:
        worst: list[tuple[Agent, int]] = [(Agent(), 100)]
        for i, p in enumerate(scores):
            if p[1] == 31:
                return self._find_losers_31(scores)
            elif p[1] < worst[0][1]:
                worst = [p]
            elif p[1] == worst[0][1]:
                worst.append(p)

        return worst

    # returns all players who don't have a score of 31
    def _find_losers_31(self, scores: list[tuple[Agent, int]]) -> list[tuple[Agent, int]]:
        losers: list[tuple[Agent, int]] = []
        for i, p in enumerate(scores):
            if p[1] != 31:
                losers.append(p)
        return losers
