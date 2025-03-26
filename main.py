from game.game import Game
from agents.human import Human
from agents.agent import Agent
from agents.random import Random
from util import score

# the most important part
# also generated from scratch by yours truly
def print_menu():
    print('''
               `'~_!*vvm9Kd$|`           
         -|VKNQQQ@@@@B@@E@@@QQNEC|:~'    
        `AQZJQ@@@@@@@N@~ !@@@@@@@@@@QQWe!
     ~*UQNQ=:Q@@@@@@QQ@@Q@@@@@@@@@@@@@@@#
 -!ugQ@@@B@@@@@@@@@@N@@@@@@@@@@@@@@@@@@@W
(Q@@@@@@@@Q@@@@@@@@@N@@@@@@@q^'Q@@@@@@@@J
b@J"Y@@@@@N@@@W;` 't@@@@@Qc`   `g@@@@@@Q!
!Q5rU@@@@@N@@N     e@@@a"       `R@@@@@q`
 |Q@@@@@@Q<@@2    ~g@Q'           k@@@@| 
  F@@@@@@>~B@Q'   |@@1             q@@g" 
  `G@@@@0  D@@@u. q@@9             *@@P  
   '%@@@'  v@@@@@dQ@@@9:  `        p@Q+  
    :B@Ql:`!@@@@@N@@@@@@@Q! -}:''rN@@N'  
     >Q@@@@Q#@@@@N@@@@@@@Rz?:N@@@@@@@C   
      v@@@@@N@@@QQ@@@@@@@@@@@@@@@@@@Q!   
      `j@@@@N@@@QQ@@@@@@@@@@@@@@N]q@b`   
       .9@@@QQ@@@QQQBBQQ@@@@@@@@m !QL    
        ,N@@@QQQQQQgDF^~:^TVHQQQ@@@&~    
         !Q@@@@@Q$|,          -~:^!-     
          ^0QNC:-                        
''')

def main():
    #run_game([Human("p1"), Human("p2")])
    run_game([Random(), Random()])

def run_game(players: list[Agent]):
    print_menu()
    winner = run_game(players)

    print(f"The winner is: {winner}")

# runs a game and returns the winner
def run_game(players: list[Agent]) -> Agent:
    while len(players) > 1:
        run_round(players)
        check_lives(players)
    
    assert len(players) == 1
    
    return players[0]

# remove players who lost all their lives
# if everyone lost, add 1 life to everyone and check again
def check_lives(players: list[Agent]):
    players_to_remove = [p for p in players if p.lives <= 0]
    if len(players_to_remove) == len(players):
        for p in players:
            p.lives += 1
        check_lives(players)
        return
    
    for p in players_to_remove:
        players.remove(p)

# runs a full round of the game
def run_round(players: list[Agent]):
    game = Game(players)
    
    while True:
        to_end = game.pick_turn()
        if to_end:
            break
    
    game.end_game()

# the first agent is now the last agent, shuffling all agents forwards
def cycle_agents(agents: list[Agent]):
    agents.append(agents.pop(0))

if __name__ == "__main__":
    main()