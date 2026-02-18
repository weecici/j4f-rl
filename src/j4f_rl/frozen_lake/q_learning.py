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
        max_exp_eps: float = 1.0,
        min_exp_eps: float = 0.05,
    ):

        exploration_eps = max_exp_eps  # Exploration rate

        # The delta for Linear decay of exploration rate
        delta_exp_eps = (max_exp_eps - min_exp_eps) / num_episodes

        for episode in range(num_episodes):
            print(f"Q-Learning start episode {episode + 1}")
            curr_state, _ = self.env.reset()
            for _ in range(max_steps):

                # Epsilon-greedy action selection
                if random.random() < exploration_eps:
                    action = random.randint(0, self.env.action_space.n - 1)
                else:
                    action = self._get_greedy_action(curr_state)

                # Take action and observe the next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                ### Update Q-table using the Q-learning update rule
                # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
                best_future_q = self.q_table[next_state].max() * (terminated == 0)
                temp_diff = (
                    reward
                    + self.gamma * best_future_q
                    - self.q_table[curr_state, action]
                )
                update_value = self.alpha * temp_diff
                self.q_table[curr_state, action] += update_value
                ###

                # Update current state
                curr_state = next_state

                if terminated or truncated:
                    break

            exploration_eps -= delta_exp_eps

    def get_action(self, state):
        return self._get_greedy_action(state)
