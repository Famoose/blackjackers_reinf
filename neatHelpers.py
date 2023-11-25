from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from blackjackenv import BlackjackEnv
import neat

# Hyperparameters
genome_rounds = 100000
hit_bonus = 0


def eval_action(action):
    if action[0] > 0.5:
        return 1
    else:
        return 0


def eval_genome(genome, config):
    fitness = 0
    agent = neat.nn.FeedForwardNetwork.create(genome, config)
    env = BlackjackEnv(sab=True)
    for i in range(0, genome_rounds):
        state, info = env.reset()
        done = False
        hitMultiplier = 1

        while not done:
            agent_action = agent.activate(state)
            # round action to 0 or 1
            agent_action = eval_action(agent_action)

            # if agent did hit give a bonus to the reward
            if agent_action == 1:
                hitMultiplier += hit_bonus

            next_state, reward, terminated, truncated, info = env.step(agent_action)

            # update if the environment is done and the current obs
            done = terminated or truncated
            state = next_state

            if (reward == 1):
                fitness += 1 * hitMultiplier

    return fitness


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        return eval_genome(self.genome, self.config)


def eval_genomes(genome, config):
    w = Worker(genome, config)
    return w.work()


def create_grids(genome, config, usable_ace=False):
    """Create policy grid given an agent."""

    # create nn feed forward network from genome
    agent = neat.nn.FeedForwardNetwork.create(genome, config)

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: eval_action(agent.activate((obs[0], obs[1], usable_ace))),
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return policy_grid


def create_plots(policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

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