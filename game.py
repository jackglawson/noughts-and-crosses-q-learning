from typing import Tuple, List
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, List
from n_and_c_settings import settings
from n_and_c_game_dependents import propagate_game, GameData
from strategy import Strategy

random.seed(2000)


class Game:
    def __init__(self, num_players: int, strategies: Union[Strategy, List[Strategy]]):
        """
        Pass a single instance of Strategy if all players are to use the same strategy.
        Pass a list of instances if each player has his own strategy.
        """
        if num_players == 1:
            assert isinstance(strategies, Strategy)
            self.strategies = strategies
            raise NotImplementedError
        else:
            if isinstance(strategies, Strategy):
                self.strategies = [strategies for _ in range(num_players)]         # will all point to the same strategy
                raise NotImplementedError
            elif isinstance(strategies, List):
                assert len(strategies) == num_players
                self.strategies = strategies
            else:
                raise Exception("invalid strategies")

        self.num_players = num_players
        self.data: GameData = GameData()
        self.log: List[GameData] = []

    def play(self, learning, explain=False, narrate=False):
        assert not self.data.winner, 'Game is already over!'

        for i in range(len(self.strategies)):
            self.strategies[i].start_new_game()

        if narrate:
            self.data.display()
        self.log.append(deepcopy(self.data))

        def player_to_play():
            p = 0
            while True:
                yield p
                p = 1 if p == 0 else 0

        player = player_to_play()
        while self.data.winner is None:
            # print(f"------ turn {self.data.turn} ------")

            p = next(player)
            action = self.strategies[p].respond(self.data, learning=learning, explain=explain)
            self.data = propagate_game(self.data, action)
            self.strategies[p].return_result(self.data, learning=learning)
            self.log.append(deepcopy(self.data))

            if narrate:
                self.data.display()

        p = next(player)
        self.strategies[p].return_result(self.data, learning=learning)
