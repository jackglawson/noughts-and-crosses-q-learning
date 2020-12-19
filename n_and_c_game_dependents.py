from dataclasses import dataclass
from typing import List, Tuple, Union
from copy import deepcopy

NUM_PLAYERS = 2
PLAYER_TOKENS = ("x", "o")
ALLOWED_ACTIONS = [(i, j) for i in range(3) for j in range(3)]


def display_board(board):
    print(" {} | {} | {} ".format(*[" " if c == "-" else c for c in board[0]]))
    print("───────────")
    print(" {} | {} | {} ".format(*[" " if c == "-" else c for c in board[1]]))
    print("───────────")
    print(" {} | {} | {} ".format(*[" " if c == "-" else c for c in board[2]]))


class GameData:
    def __init__(self):
        self.winner: Union[None, str] = None
        self.turn: int = 1
        self.board = [["-", "-", "-"] for _ in range(3)]

    def __repr__(self):
        return str(vars(self))

    def display(self):
        display_board(self.board)


@dataclass(frozen=True)
class PureState:
    board: List[List[str]]

    @classmethod
    def build_from_data(cls, data: GameData):
        return cls(data.board)

    def display(self):
        display_board(self.board)

    def __repr__(self):
        return str(self.board)

    def __hash__(self):
        return hash(str(self.board))


def action_is_valid(board: List[List[str]], action: Tuple[int, int]) -> bool:
    return True if board[action[0]][action[1]] == "-" else False


def get_allowed_actions(pure_state: PureState) -> List[Tuple[int, int]]:
    return [action for action in ALLOWED_ACTIONS if action_is_valid(pure_state.board, action)]


def player_to_play(game_data: GameData):
    return (game_data.turn - 1) % 2


def propagate_game(initial_game_data: GameData, action: Tuple[int, int]) -> GameData:
    """
    Play a single turn and update the game data.
    Action is a tuple describing the location of play.
    """

    assert action_is_valid(initial_game_data.board, action), "Invalid move!"

    final_game_data = deepcopy(initial_game_data)

    # crosses always start
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

    # check for draw
    if final_game_data.turn == 9 and not final_game_data.winner:
        final_game_data.winner = "draw"

    final_game_data.turn += 1

    return final_game_data


def get_reward(initial_data: GameData, final_data: GameData) -> float:
    """Calculate the reward associated with the last move"""
    player_just_played = PLAYER_TOKENS[(initial_data.turn - 1) % NUM_PLAYERS]

    if final_data.winner is None or final_data.winner == "draw":
        reward = 0
    elif final_data.winner == player_just_played:
        reward = 1
    else:
        reward = -1

    return reward


def request_move_from_user(game_data: GameData):
    game_data.display()
    user_input = input("Please input move: ")
    return tuple(user_input)
