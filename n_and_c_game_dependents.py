from n_and_c_settings import settings

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
from copy import deepcopy
import random


class GameData:
    def __init__(self):
        self.winner: Union[None, str] = None
        self.turn: int = 1
        self.board = [["-", "-", "-"] for _ in range(3)]

    def __repr__(self):
        return str(vars(self))


def propagate_game(initial_game_data: GameData, action: Tuple[int, int]) -> GameData:
    """
    Play a single turn and update the game data.
    Action is a tuple describing the location of play.
    """

    final_game_data = deepcopy(initial_game_data)

    current_player = 'o' if initial_game_data.turn % 2 == 0 else 'x'
    final_game_data.board[action[0]][action[1]] = current_player

    # check for win
    board = final_game_data.board
    for player in ['o', 'x']:
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == player or board[0][i] == board[1][i] == board[2][i] == player:
                final_game_data.winner = player
        if board[0][0] == board[1][1] == board[2][2] == player or board[2][0] == board[1][1] == board[0][2] == player:
            final_game_data.winner = player

    return final_game_data


def get_reward(initial_data: GameData, final_data: GameData) -> float:
    """Calculate the reward associated with the last move"""
    if final_data.winner is None:
        reward = 0
    elif final_data.winner == settings.ai_plays_as:
        reward = 1
    else:
        reward = -1

    return reward


@dataclass(frozen=True)
class PureState:
    board: List[List[str]]

    @classmethod
    def build_from_data(cls, data: GameData):
        return cls(data.board)

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        return hash(str(self.board))


START_Q = 0
ALLOWED_ACTIONS = [(i, j) for i in range(3) for j in range(3)]
