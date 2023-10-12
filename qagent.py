from blackjackenv import BlackjackEnv
from collections import defaultdict

import numpy as np


env = BlackjackEnv(sab=True)


class QAgent:
    def __init__(
            self,
            exp_policy: str,  # greedy, e_greedy, softmax
            alpha: float,  # learning rate
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            gamma: float,  # discount factor
    ):

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.exploration_policy = exp_policy

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, state: tuple[int, int, bool]) -> int:
        if self.exploration_policy == "greedy":
            return int(np.argmax(self.q_values[state]))

        elif self.exploration_policy == "e_greedy":
            if np.random.random() < self.epsilon:
                return env.action_space.sample()
            else:
                return int(np.argmax(self.q_values[state]))

        elif self.exploration_policy == "softmax":
            if np.random.random() < self.epsilon:
                return env.action_space.sample()
            else:
                return int(np.argmax(np.exp(self.q_values[state]) / np.sum(np.exp(self.q_values[state]))))

    def update(
            self,
            state: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_state: tuple[int, int, bool],
    ):

        # Update Q-value of an action
        old_values = self.q_values[state][action]
        next_max_values = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = (
                reward + self.gamma * next_max_values - old_values
        )
        new_values = old_values + self.alpha * temporal_difference

        # update the q_values
        self.q_values[state][action] = new_values

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)