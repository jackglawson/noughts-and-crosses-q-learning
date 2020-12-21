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
    def __init__(self):
        Strategy.__init__(self)

    def start_new_game(self):
        pass

    def respond(self, game_data: GameData, **kwargs):
        return request_move_from_user(game_data)

    def return_result(self, final_data: GameData, **kwargs):
        pass


class RandomStrategy(Strategy):
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
        learning = kwargs["learning"]
        explain = kwargs["explain"] if "explain" in kwargs else False

        pure_state = PureState.build_from_data(game_data)

        try:
            state = self.states[pure_state]
        except KeyError:
            state = State(pure_state, self.p)
            self.states[pure_state] = state

        if learning and self.p.predictive and self.last_state is not None:
            self.last_state.update_max_q_values_of_next_states(state, self.last_action)

        if learning:
            action = state.explore(explain=explain)
        else:
            action = state.exploit(explain=explain)

        self.last_state = state
        self.last_action = action
        self.last_data = game_data

        return action

    def return_result(self, final_data, **kwargs):
        """The game data immediately after the previous action will be returned here to allow learning"""
        learning = kwargs["learning"]
        if learning:
            reward = get_reward(self.last_data, final_data)
            self.last_state.update_q_value(self.last_action, reward)


@dataclass(frozen=False)
class State:
    """Q-values are stored here. Each PureState has its own State."""
    pure_state: PureState
    p: StrategyParams

    def __post_init__(self):
        self.allowed_actions = get_allowed_actions(self.pure_state)
        self.actions = dict([(action, self.p.start_q) for action in self.allowed_actions])
        self.max_q_values_of_next_states = dict([(action, np.nan) for action in self.allowed_actions])
        self.total_hits = 0
        self.num_hits = dict([(action, 0) for action in self.allowed_actions])
        self.history = []
        self.past_learned_values = dict([(action, []) for action in self.allowed_actions])
        self.last_decision_type = None

    def explore(self, explain=False):
        """
        Choose an action. If the number of hits on any of the possible actions is less than
        p.min_hits_before_using_stats, a random action is chosen. Else, choose an action based on q-values
        """

        if any([hit < self.p.min_hits_before_using_stats for hit in self.num_hits.values()]):
            action = random.choice(list(self.actions.keys()))
            if explain:
                print('Exploring the state:')
                print(self.pure_state)
                print('Not enough hits, choosing at random')
                print('Action to take: {}'.format(action))

        elif random.random() < self.p.random_action_rate:
            action = random.choice(list(self.actions.keys()))
            if explain:
                print('Exploring the state:')
                print(self.pure_state)
                print('Enough hits but choosing at random')
                print('Action to take: {}'.format(action))

        else:
            # Take a random sample of the past learned q-values for each action. The action corresponding to the
            # highest sample is chosen. This way, actions with higher q-values get chosen preferentially.
            random_choices_from_past = dict(
                [(action, random.choice(past_values[-self.p.max_hits_used_in_stats:])) for action, past_values in
                 self.past_learned_values.items()])
            action = max(random_choices_from_past, key=random_choices_from_past.get)

            if explain:
                print('Exploring the state:')
                print(self.pure_state)
                print('Actions: {}'.format(self.actions))
                print('Hits: {}'.format(self.num_hits))
                print('Random samples chosen: {}'.format(random_choices_from_past))
                print('Action to take: {}'.format(action))
                plt.hist([self.past_learned_values[action_][-self.p.max_hits_used_in_stats:] for action_ in
                          self.allowed_actions], label=self.allowed_actions)
                plt.legend()
                plt.show()

        self.last_decision_type = 'explore'

        return action

    def exploit(self, explain=False):
        """Choose the action with the highest q-value"""
        action = max(self.actions, key=self.actions.get)
        self.last_decision_type = 'exploit'

        if explain:
            print('Exploiting the state:')
            print(self.pure_state)
            print('Action to take: {}'.format(action))
            for action_ in self.actions.keys():
                selected_history = [hist for hist in self.history if hist['action'] == action_]
                plt.plot([hist['hit'] for hist in selected_history], [hist['q_value'] for hist in selected_history],
                         label=action_)
            plt.legend()
            plt.show()

        return action

    def update_q_value(self, action, reward):
        self.total_hits += 1
        self.num_hits[action] += 1

        if not self.p.predictive or np.isnan(self.max_q_values_of_next_states[action]):
            learned_value = reward
        else:
            learned_value = reward + self.p.discount_rate * self.max_q_values_of_next_states[action]

        self.actions[action] = (((self.num_hits[action] - 1) * self.actions[action]) + learned_value) / (self.num_hits[action])

        self.history.append(deepcopy({'hit': self.num_hits[action],
                                      'action': action,
                                      'reward': reward,
                                      'learned_value': learned_value,
                                      'q_value': self.actions[action],
                                      'decision_type': self.last_decision_type,
                                      }))
        if self.p.predictive:
            self.history[-1]['max_q_of_next_state'] = self.max_q_values_of_next_states

        self.past_learned_values[action].append(learned_value)

    def update_max_q_values_of_next_states(self, next_state, last_action):
        learned_value = max(next_state.actions.values())
        if self.p.next_state_is_predictable or np.isnan(self.max_q_values_of_next_states[last_action]):
            self.max_q_values_of_next_states[last_action] = learned_value
        else:
            self.max_q_values_of_next_states[last_action] = (((self.total_hits - 1) * self.max_q_values_of_next_states[
                last_action]) + learned_value) / self.total_hits
