from blackjackenv import BlackjackEnv
from collections import defaultdict
from typing import Optional
from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns
from matplotlib.patches import Patch


env = BlackjackEnv(sab=True)


class QAgent:
    def __init__(
            self,
            alpha: float,  # learning rate
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            gamma: float,  # discount factor
            exp_policy: Optional[str] = "e_greedy",  # greedy, e_greedy, softmax
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


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig