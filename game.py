from typing import Tuple, List
import random
from copy import deepcopy
from dataclasses import dataclass
from n_and_c_settings import settings
from n_and_c_game_dependents import propagate_game, GameData

random.seed(2000)


class Game:
    def __init__(self, strategy):
        self.strategy = strategy
        self.data: GameData = GameData()
        self.log: List[GameData] = []

    def play(self):
        assert not self.data.winner, 'Game is already over!'
        self.strategy.start_new_game()

        if settings.narrate_game:
            print(self.data)
        self.log.append(deepcopy(self.data))

        while not self.data.winner:
            action = self.strategy.respond(self.data)
            self.data = propagate_game(self.data, action)
            self.strategy.return_result(self.data)
            self.log.append(deepcopy(self.data))

            if settings.narrate_game:
                print(self.data)
