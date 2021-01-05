from n_and_c_game_dependents import get_allowed_actions, get_reward, PureState, GameData, request_move_from_user
from n_and_c_params import StrategyParams

from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


class Strategy:
    def __init__(self):
        pass

    def start_new_game(self):
        raise NotImplementedError

    def respond(self, game_data: GameData, **kwargs):
        raise NotImplementedError

    def return_result(self, final_data: GameData, **kwargs):
        raise NotImplementedError


class UserInput(Strategy):
    """Allows the user to input moves"""
    def __init__(self):
        Strategy.__init__(self)

    def start_new_game(self):
        pass

    def respond(self, game_data: GameData, **kwargs):
        return request_move_from_user(game_data)

    def return_result(self, final_data: GameData, **kwargs):
        pass


class RandomStrategy(Strategy):
    """Chooses actions randomly"""
    def __init__(self):
        Strategy.__init__(self)

    def start_new_game(self):
        pass

    def respond(self, game_data: GameData, **kwargs):
        pure_state = PureState(game_data.board)
        return random.choice(get_allowed_actions(pure_state))

    def return_result(self, final_data: GameData, **kwargs):
        pass


class LearningStrategy(Strategy):
    """Q-learning strategy"""
    def __init__(self, p: StrategyParams):
        Strategy.__init__(self)
        self.p = p
        self.states = {}
        self.last_state = None
        self.last_action = None
        self.last_data = None

    def start_new_game(self):
        self.last_state = None
        self.last_action = None
        self.last_data = None

    def respond(self, game_data: GameData, **kwargs):
        """Respond to game data with an action"""
        explain = kwargs["explain"] if "explain" in kwargs else False

        pure_state = PureState.build_from_data(game_data)

        try:
            state = self.states[pure_state]
        except KeyError:
            state = State(pure_state, self.p)
            self.states[pure_state] = state

        if self.p.learning and self.p.predictive and self.last_state is not None:
            self.last_state.update_max_q_values_of_next_states(state, self.last_action)

        if not self.p.learning:
            action = state.exploit(explain=explain)
        else:
            if random.random() < state.epsilon:
                action = state.explore(explain=explain)
            else:
                action = state.exploit(explain=explain)

        self.last_state = state
        self.last_action = action
        self.last_data = game_data

        return action

    def return_result(self, final_data, **kwargs):
        """The game data immediately after the previous action will be returned here to allow learning"""
        if self.p.learning:
            reward = get_reward(self.last_data, final_data)
            self.last_state.update_q_value(self.last_action, reward)


@dataclass(frozen=False)
class State:
    """Q-values are stored here. Each PureState has its own State."""
    pure_state: PureState
    p: StrategyParams

    def __post_init__(self):
        self.allowed_actions = get_allowed_actions(self.pure_state)
        self.q_values = dict([(action, self.p.start_q) for action in self.allowed_actions])
        self.max_q_values_of_next_states = dict([(action, np.nan) for action in self.allowed_actions])
        self.total_hits = 0
        self.num_hits = dict([(action, 0) for action in self.allowed_actions])
        self.epsilon = 1
        self.last_decision_type = None
        if self.p.keep_log:
            self.history = []

    def explore(self, explain=False):
        """Choose an action at random"""

        action = random.choice(self.allowed_actions)
        self.last_decision_type = 'explore'

        if explain:
            print('Exploring the state:')
            print(self.pure_state)
            print('Choosing at random')
            print('Action to take: {}'.format(action))

        return action

    def exploit(self, explain=False):
        """Choose the action with the highest q-value"""
        max_q = max(self.q_values.values())
        best_actions = filter(lambda a: self.q_values[a] == max_q, self.q_values.keys())
        action = random.choice(list(best_actions))
        self.last_decision_type = 'exploit'

        if explain:
            print('Exploiting the state:')
            print(self.pure_state)
            print('Action to take: {}'.format(action))
            self.plot()

        return action

    def update_q_value(self, action, reward):
        self.total_hits += 1
        self.epsilon = max(self.epsilon * self.p.epsilon_decay_rate, self.p.minimum_epsilon)
        self.num_hits[action] += 1

        if not self.p.predictive or np.isnan(self.max_q_values_of_next_states[action]):
            learned_value = reward
        else:
            learned_value = reward + self.p.discount_rate * self.max_q_values_of_next_states[action]

        self.q_values[action] = self.q_values[action] + self.p.learning_rate * (learned_value - self.q_values[action])

        if self.p.keep_log:
            self.history.append(deepcopy({'hit': self.num_hits[action],
                                          'action': action,
                                          'reward': reward,
                                          'learned_value': learned_value,
                                          'q_value': self.q_values[action],
                                          'q_values': self.q_values,
                                          'decision_type': self.last_decision_type,
                                          }))
            if self.p.predictive:
                self.history[-1]['max_q_of_next_state'] = self.max_q_values_of_next_states

    def update_max_q_values_of_next_states(self, next_state, last_action):
        learned_value = max(next_state.q_values.values())
        if self.p.next_state_is_predictable or np.isnan(self.max_q_values_of_next_states[last_action]):
            self.max_q_values_of_next_states[last_action] = learned_value
        else:
            self.max_q_values_of_next_states[last_action] = (((self.total_hits - 1) * self.max_q_values_of_next_states[
                last_action]) + learned_value) / self.total_hits

    def plot(self):
        assert self.p.keep_log, "Can't plot without history"
        for action_ in self.q_values.keys():
            selected_history = [hist for hist in self.history if hist['action'] == action_]
            plt.plot([hist['hit'] for hist in selected_history], [hist['q_value'] for hist in selected_history],
                     label=action_)
        plt.legend()
        plt.show()
