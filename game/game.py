import random
from agents.agent import Agent
from game.card import Card
from game.card import Suit
from game.action import Action
from util import score

class Game:
    def __init__(self, players: list[Agent]):
        self.players: list[Agent] = players
        self.deck: list[Card] = self._create_deck()
        self.discard: list[Card] = []
        self.turn: int = 0
        self.first_turn = True

        # represents who called the game
        self.called: int = -1

        self._build_hands()
    
    # builds 52 card deck
    def _create_deck(self) -> list[Card]:
        deck = []

        for e in Suit:
            # number cards
            for i in range(2, 10):
                deck.append(Card(i, e))
            
            # face cards
            for _ in range(0, 4):
                deck.append(Card(10, e))
            # ace
            deck.append(Card(11, e))

        random.shuffle(deck)
        return deck
    
    # puts 3 cards from the deck in each person's hands
    def _build_hands(self):
        assert len(self.players) < len(self.deck) // 3

        for player in self.players:
            player.hand = [self.deck.pop() for _ in range(0, 3)]
    
    # the main game runner that picks the turn for the agent to take
    # if the game is ended, returns true
    def pick_turn(self) -> bool:
        # TODO: rework how stuff gets handed off to players
        agent = self.players[self.turn]
        
        action: tuple[Action, int] = ()
        if self.first_turn:
            action = agent.take_first_turn()
            self.first_turn = False
            if action[0] == Action.CALL:
                return True
        elif game.called != -1:
            action = agent.take_turn()
        else:
            action = agent.take_last_turn()
            assert action[0] != Action.CALL
        
        self._take_action(action)
        self._next_turn()

        # if the game is ending normally
        return self.called == self.turn

    # takes in a tuple containing the actiont to take and what card to replace
    # in the case of a call, the second int's value doesn't matter
    def _take_action(self, action: tuple[Action, int]):
        match action[0]:
            case Action.CALL:
                self.called == self.turn
            case Action.DRAW:
                assert len(self.deck) >= 1

                self.discard.push(self.players[self.turn].hand[action[1]])
                self.players[self.turn].hand[action[1]] = self.deck.pop()
                self.discard[-1].visible = True
            case Action.DRAW_DISCARD:
                assert len(self.discard) >= 1

                discarded = self.players[self.turn].hand[action[1]]
                self.players[self.turn].hand[action[1]] = self.discard.pop()
                self.discard.push(discarded)
                self.discard[-1].visible = True
            case _:
                raise ValueError("Unknown action taken")

    # increments and wraps the turn counter
    def _next_turn(self):
        assert self.turn >= 0
        
        self.turn += 1
        self.turn %= len(players)


    # ends the game for a normal ending
    # the person with the least score loses a life
    # if the person who called is least, two lives are lost
    def end_game(self):
        assert self.called != -1
        
        losers = self._find_losers()
        for loser in losers:
            if loser == self.players[self.called]:
                loser.lives -= 2
            else:
                loser.lives -= 1


    # returns a list (handling ties) contining tuples of (Agent, score) of those who lost
    def _find_losers(self) -> list[tuple[Agent, int]]:
        worst: list[tuple[Agent, int]] = [(None, 0)]
        
        for p in self.players:
            s = score.score(p.hand)

            # break out of regular scoring and return the 31 case
            if s == 31:
                return self._find_losers_31()

            if s < worst[0][1]:
                worst = [(p, s)]
            elif s == worst[0][1]:
                worst.append((p, s))
        
        assert worst[0][0] != None

        return worst
    
    # returns all players who don't have a score of 31
    def _find_losers_31(self) -> list[tuple[Agent, int]]:
        losers: list[tuple[Agent, int]] = []
        for p in self.players:
            if score.score(p.hand) != 31:
                losers.append((p, s))
        
        return losers
