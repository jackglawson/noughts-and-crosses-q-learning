from strategy import LearningStrategy, UserInput
from n_and_c_params import sp_learning_x, sp_learning_o, sp_random, p
from n_and_c_game_dependents import NUM_PLAYERS
from game import Game

import time
import pickle


def save(obj, filename: str):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)
    print('Successfully saved to {}'.format(filename))


if __name__ == "__main__":
    s = time.time()

    strategy_x = LearningStrategy(sp_learning_x)
    strategy_o = LearningStrategy(sp_learning_o)
    strategies = [strategy_x, strategy_o]

    for epoch in range(p.num_epochs):
        game = Game(NUM_PLAYERS, strategies)
        game.play(learning=p.learning, explain=p.explain)
        if epoch % 1000 == 0:
            print('Progress: {}%'.format(round(epoch/p.num_epochs*100), 3))

    save(strategies, "learning_strategies.pkl")

    print("Time taken: {} seconds".format(round(time.time() - s, 1)))
