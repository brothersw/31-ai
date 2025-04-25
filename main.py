from matplotlib import pyplot as plt
from agents.ai import AIAgent
from game.game import Game
from agents.human import Human
from agents.agent import Agent
from agents.random import Random
from agents.greedy import Greedy
from util import score
import os

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
    players = [AIAgent(), Human("p2")]
    players[0].load_checkpoint("./200_checkpoint.pt", epsilon=0.0)
    host_game(players)
    
    #run_training([AIAgent(), Greedy(), Random()])


def host_game(players: list[Agent]):
    print_menu()
    winner = run_game(players)

    print(f"The winner is: {winner}")

# runs a game and returns the winner
def run_game(players: list[Agent]) -> Agent:
    while len(players) > 1:
        run_round(players, True)
        check_lives(players)
        cycle_agents(players)
        print("End of round: ")
        print(players)
    
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

def run_training(players: list[Agent]):
    print_menu()
    active_players = players.copy()
    wins = [0 for i in range(len(players))]
    global_wins = wins.copy()

    win_history = [[] for _ in players]
    
    batches = 400
    for i in range(batches):
        for _ in range(1000):
            # Get players with lives remaining
            active_players = [p for p in active_players if p.lives > 0]
            
            # If only 1 player remains, revive everyone and distribute wins
            if len(active_players) <= 1:
                assert len(active_players) == 1

                wins[players.index(active_players[0])] += 1
                global_wins[players.index(active_players[0])] += 1

                for player in players:
                    player.revive()
                    active_players = players.copy()
            
            run_round(active_players, False)
            cycle_agents(active_players)
        
        print(f"Wins for batch {i}: {wins}")
        
        if i % 50 == 0:
            print(f"saving checkpoint {i}_checkpoint.pt")
            players[0].save_checkpoint(f"./{i}_checkpoint.pt")

        for j, w in enumerate(wins):
            win_history[j].append(w)
        
        wins = [0] * len(players)
        
    
    print(f"Global wins: {global_wins}")

    #players[0].plot_metrics()
    players[0].plot_rolling_average()
    
    for i, w in enumerate(win_history):
        plt.plot(range(batches), w, label=f"Player {i} {type(players[i])}")
    
    plt.xlabel('Batch # (1 batch = 1000 rounds)')
    plt.ylabel('Wins/Batch')
    plt.grid(True)
    plt.legend()
    plt.show()


# runs a full round of the game
def run_round(players: list[Agent], printing: bool):
    game = Game(players)
    
    while True:
        to_end = game.pick_turn(print_state=printing)
        if to_end:
            break
    
    game.end_game()

# the first agent is now the last agent, shuffling all agents forwards
def cycle_agents(agents: list[Agent]):
    agents.append(agents.pop(0))

if __name__ == "__main__":
    main()
