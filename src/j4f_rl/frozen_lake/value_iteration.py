import gymnasium as gym


class Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        eps: float = 1e-9,
    ):
        self.env = env.unwrapped
        self.value_table = [0 for _ in range(env.observation_space.n)]
        self.policy = [0 for _ in range(env.observation_space.n)]
        self.gamma = gamma  # Discount factor
        self.eps = eps  # Convergence threshold

    def value_iteration(self):
        count = 1

        # Iterate until convergence
        while count > 0:
            count = 0

            # Update value table and policy
            for state in range(self.env.observation_space.n):
                best_future_rewards = 0
                best_action = -1

                # Find best action in current state
                for action in range(self.env.action_space.n):
                    future_rewards = 0

                    # Calculate expected future rewards
                    # self.env.P is in form of (probability, next_state, reward, done)
                    for prob, next_state, _, _ in self.env.P[state][action]:
                        future_rewards += prob * self.value_table[next_state]

                    # Update best action if found better one
                    if future_rewards > best_future_rewards:
                        best_future_rewards = future_rewards
                        best_action = action

                old_value = self.value_table[state]

                # Update value table and policy
                self.value_table[state] = (
                    float(state + 1 == self.env.observation_space.n)
                    + self.gamma * best_future_rewards
                )
                self.policy[state] = best_action

                # Check for convergence
                if self.value_table[state] - old_value > self.eps:
                    count += 1

    def get_action(self, state):
        return self.policy[state]
