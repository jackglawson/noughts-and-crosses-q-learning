from params import Params, StrategyParams


p = Params(num_epochs=100000,
           explain=False,
           learning=True)

# Strategy for x
sp_learning_x = StrategyParams(start_q=0.3067,
                               random_action_rate=0.3,
                               discount_rate=0.7,
                               min_hits_before_using_stats=20,
                               max_hits_used_in_stats=100,
                               next_state_is_predictable=False,
                               predictive=True)

# Strategy for o
sp_learning_o = StrategyParams(start_q=-0.3067,
                               random_action_rate=0.3,
                               discount_rate=0.7,
                               min_hits_before_using_stats=20,
                               max_hits_used_in_stats=100,
                               next_state_is_predictable=False,
                               predictive=True)
# Start_q has been chosen as such because even with a random strategy, x wins more often than o.
# If start_q is set at 0 for both players, moves that are good at the start of learning when the play is random (e.g.
# placing first x in center) get over-valued and other potentially good moves are neglected.


# A strategy that only chooses random actions.
sp_random = StrategyParams(start_q=0,
                           random_action_rate=1,
                           discount_rate=0.7,
                           min_hits_before_using_stats=1,
                           max_hits_used_in_stats=1,
                           next_state_is_predictable=False,
                           predictive=True)
