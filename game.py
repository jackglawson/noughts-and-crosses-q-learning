from n_and_c_game_dependents import propagate_game, GameData, player_to_play

from copy import deepcopy
from typing import List
from strategy import Strategy


class Game:
    """
    The game is played here. Game keeps track of GameData, asks Strategy for the next move, propagates the game,
    and finally returns the resulting game_data back to Strategy.
    """
    def __init__(self, num_players: int, strategies: List[Strategy]):
        assert len(strategies) == num_players
        self.strategies = strategies
        self.num_players = num_players
        self.data: GameData = GameData()
        self.log: List[GameData] = []

    def play(self, learning, explain=False, narrate=False):
        assert self.data.winner is None, 'Game is already over!'

        for i in range(len(self.strategies)):
            self.strategies[i].start_new_game()

        if narrate:
            self.data.display()
        self.log.append(deepcopy(self.data))

        while self.data.winner is None:
            p = player_to_play(self.data)
            action = self.strategies[p].respond(self.data, learning=learning, explain=explain)
            self.data = propagate_game(self.data, action)
            self.strategies[p].return_result(self.data, learning=learning)
            self.log.append(deepcopy(self.data))

            if narrate:
                self.data.display()

        p = player_to_play(self.data)
        self.strategies[p].return_result(self.data, learning=learning)

        if narrate:
            if self.data.winner == "draw":
                print("It's a draw!")
            else:
                print("{}'s win!".format(self.data.winner))
