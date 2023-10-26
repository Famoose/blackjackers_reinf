import numpy as np
from tqdm import tqdm
import gymnasium
class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state) -> int:
        return self.env.action_space.sample()


def playNTimes(env, agent, N):

    env = gymnasium.wrappers.RecordEpisodeStatistics(env, deque_size=N)

    # For plotting metrics
    timesteps_per_episode = []
    wins = []

    for i in tqdm(range(0, N)):

        # reset environment to a random state
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        terminated = False

        while not terminated:

            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            if reward == -1:
                penalties += 1

            state = next_state
            epochs += 1

        timesteps_per_episode.append(epochs)
        wins.append(reward)
    return env, timesteps_per_episode, wins

def calculate_winrate(env):
    return np.sum(np.array(env.return_queue).flatten() == 1) / len(env.return_queue)
