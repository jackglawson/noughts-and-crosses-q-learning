from params import Params, StrategyParams

p = Params(num_epochs=50000,
           explain=False)

# Strategy for x
sp_learning_x = StrategyParams(start_q=0.3,
                               learning_rate=0.1,
                               discount_rate=0.2,
                               epsilon_decay_rate=0.0,
                               minimum_epsilon=0.4,
                               next_state_is_predictable=False,
                               predictive=True,
                               learning=True,
                               keep_log=True)

# Strategy for o
sp_learning_o = StrategyParams(start_q=-0.3,
                               learning_rate=0.1,
                               discount_rate=0.2,
                               epsilon_decay_rate=0.0,
                               minimum_epsilon=0.4,
                               next_state_is_predictable=False,
                               predictive=True,
                               learning=True,
                               keep_log=True)

