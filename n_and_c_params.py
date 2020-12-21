from params import Params, StrategyParams


p = Params(num_epochs=10000,
           explain=False,
           learning=True)

# Strategy for x
sp_learning_x = StrategyParams(start_q=0,
                               random_action_rate=0.4,
                               discount_rate=0.7,
                               next_state_is_predictable=False,
                               predictive=True)

# Strategy for o
sp_learning_o = StrategyParams(start_q=0,
                               random_action_rate=0.4,
                               discount_rate=0.7,
                               next_state_is_predictable=False,
                               predictive=True)

# A strategy that only chooses random actions.
sp_random = StrategyParams(start_q=0,
                           random_action_rate=1,
                           discount_rate=0.7,
                           next_state_is_predictable=False,
                           predictive=True)
