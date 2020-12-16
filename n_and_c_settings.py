from dataclasses import dataclass
from typing import Tuple


@dataclass
class Settings:
    pass


@dataclass
class LearningParams:
    # random_action_rate: float = 0.5
    discount_rate: float = 0.9              # A reward now is better than future rewards.
                                            # The larger the factor, more weight will be given to future rewards.
                                            # Set discount_rate = 1 to give equal weight to immediate and future rewards.
    num_epochs: int = 100000
    explain: bool = False
    learning: bool = True
    # explore_multiplier: float = 1.0       # increasing this will make it more likely to choose action at random
    next_state_is_predictable = False       # unpredictable because there are two players
    min_hits_before_using_stats = 100000
    max_hits_used_in_stats = 100
    predictive = True                  # if True, strategy will use max q of next state. Should be True if the reward is not given immediately


settings = Settings()
learning_params = LearningParams()
