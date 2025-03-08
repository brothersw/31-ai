from game.game import Game
from agents.human import Human
from agents.agent import Agent
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
    print_menu()

    run_game([Human("p1"), Human("p2")])

def run_game(players: list[Agent]):
    while len(players) >= 1:
        run_round(players) 
        
        # TODO: doesn't handle ties that end in everybody losing all of their lives
        for player in players:
            if player.lives <= 0:
                players.remove(player)

    assert len(players) == 1
    print(f"The winner is: {players[0]}")

# runs a full round of the game
def run_round(players: list[Agent]):
    game = Game(players)
    
    while True:
        to_end = game.pick_turn()
        if to_end:
            break

    print(game.state.deck)
    print(game.state.discard)
    for p in game.state.players:
        print(p.hand)
        print(score.score(p.hand))

    game.end_game()

if __name__ == "__main__":
    main()
