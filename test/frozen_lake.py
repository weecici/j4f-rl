import gymnasium as gym
import ale_py
from j4f_rl.frozen_lake import FrozenLakeAgent

gym.register_envs(ale_py)
env = gym.make(
    "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=True, render_mode="human"
)


if __name__ == "__main__":

    observation, info = env.reset()
    agent = FrozenLakeAgent(env)
    agent.value_iteration()

    for _ in range(1000):
        action = agent.get_action(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
