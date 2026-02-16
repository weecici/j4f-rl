import gymnasium as gym
import random
import numpy as np

random.seed(42)


class Agent:
    def __init__(
        self, env: gym.Env, alpha: float = 0.1, gamma: float = 0.99, eps: float = 1e-9
    ):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        # self.policy = np.zeros((env.observation_space.n), dtype=np.int8)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.eps = eps  # Convergence threshold

    def _get_greedy_action(self, state):
        q_vals = self.q_table[state]
        best_indices = np.flatnonzero(q_vals == q_vals.max())
        return int(random.choice(best_indices))

    def learn(
        self,
        num_episodes: int = 100,
        max_steps: int = 100,
        exploration_eps: float = 1.0,
        decay_eps: float = 0.99,
    ):
        for episode in range(num_episodes):
            print(f"Q-Learning start episode {episode + 1}")
            state, _ = self.env.reset()
            for _ in range(max_steps):
                if random.random() < exploration_eps:
                    action = random.randint(0, self.env.action_space.n - 1)
                else:
                    action = self._get_greedy_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                best_future_q = self.q_table[next_state].max() * (terminated == 0)
                temp_diff = (
                    reward + self.gamma * best_future_q - self.q_table[state, action]
                )

                update_value = self.alpha * temp_diff
                self.q_table[state, action] += update_value

                state = next_state

                if terminated or truncated:
                    break

            exploration_eps *= decay_eps

    def get_action(self, state):
        return self._get_greedy_action(state)
