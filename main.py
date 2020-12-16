from strategy import Strategy
from n_and_c_settings import learning_params
from n_and_c_game_dependents import NUM_PLAYERS
from game import Game

import time
import pickle

start = time.time()


def save(learning_strategies, last_game):
    end = time.time()

    with open('learning_strategies.pkl', 'wb') as pickle_file:
        pickle.dump(learning_strategies, pickle_file)

    with open('last_game.pkl', 'wb') as pickle_file:
        pickle.dump(last_game, pickle_file)

    print('Time taken: {}'.format(end - start))
    print('SAVED')


strategy_x = Strategy()
strategy_o = Strategy()
strategies = [strategy_x, strategy_o]


for epoch in range(learning_params.num_epochs):
    # print('#################### GAME {} ##################'.format(epoch))
    game = Game(NUM_PLAYERS, strategies)
    game.play(learning=learning_params.learning, explain=learning_params.explain)
    last_game = game
    if epoch % (1000) == 0:
        print('Num epochs: {}'.format(epoch))
        #print('Progress: {}%'.format(epoch/learning_params.num_epochs*100))


save(strategies, last_game)


#animate(game.log)