from dataclasses import dataclass


@dataclass()
class Params:
    num_epochs: int
    explain: bool
    learning: bool


@dataclass
class StrategyParams:
    """
    Parameters for Strategy.

    Parameters
    ----------
    start_q: float
      Each action has an associated q-value. The q-values will be initialised at start_q.

    random_action_rate: float
      Even if the number of hits is over min_hits_before_using_stats, this parameter forces some exploration.

    discount_rate: float
      A reward now is better than future rewards.
      The larger the factor, more weight will be given to future rewards.
      Set discount_rate = 1 to give equal weight to immediate and future rewards.

    min_hits_before_using_stats: int
      Moves are chosen according to how successful they were in the past.
      To ensure good exploration, actions should be chosen randomly for a while before using past actions to decide.

    max_hits_used_in_stats: int
      We don't want to take the whole history of actions into account when deciding which moves are good,
      because some will be using out-of-date strategy.

    next_state_is_predictable: bool
      Is the next state purely dependent on the action now?
      Should be false if there is randomness in the game, or there is more than one player.

    predictive: bool
      if True, strategy will use max q of next state. Should be True if the reward is not given immediately.
    """

    start_q: float
    random_action_rate: float
    discount_rate: float
    min_hits_before_using_stats: int
    max_hits_used_in_stats: int
    next_state_is_predictable: bool
    predictive: bool


p = Params(num_epochs=1000000,
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
