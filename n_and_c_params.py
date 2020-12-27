from params import Params, StrategyParams

p = Params(num_epochs=10000,
           explain=False)

# Strategy for x
sp_learning_x = StrategyParams(start_q=0.3,
                               random_action_rate=0.4,
                               discount_rate=0.7,
                               min_hits_before_exploit=10,
                               next_state_is_predictable=False,
                               predictive=True,
                               learning=True,
                               keep_log=True)

# Strategy for o
sp_learning_o = StrategyParams(start_q=-0.3,
                               random_action_rate=0.4,
                               discount_rate=0.7,
                               min_hits_before_exploit=10,
                               next_state_is_predictable=False,
                               predictive=True,
                               learning=True,
                               keep_log=True)

