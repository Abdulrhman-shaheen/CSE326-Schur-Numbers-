import gym
import numpy as np
from gym import spaces

class SchurEnv(gym.Env):
    def __init__(self, max_z=192, initial_seq=[(1,1), (2,2), (3,2), (4,1), (5,3)]):
        super().__init__()
        self.max_z = max_z
        self.initial_seq = initial_seq
        self.reset()
        self.action_space = spaces.Discrete(5)  # Colors 0-4 (map to 1-5)
        self.observation_space = spaces.Box(
            low=0, high=max_z, shape=(200, 2), dtype=np.int32
        )

    def reset(self):
        self.z = len(self.initial_seq) + 1
        self.pairs = self.initial_seq.copy()
        return self._get_obs()

    def step(self, action):
        color = action + 1  # Map action 0-4 to color 1-5
        reward = 0
        done = False
        info = {}

        if self.is_valid(color, self.z):
            self.pairs.append((self.z, color))
            self.z += 1
            reward = 1.0
            if self.z > self.max_z:
                done = True
        else:
            reward = -10.0
            done = True
            info["reason"] = "Invalid color"

        obs = self._get_obs()
        return obs, reward, done, info

    def is_valid(self, color, z):
        color_nums = [p[0] for p in self.pairs if p[1] == color] + [z]
        for x in range(1, z // 2 + 1):
            if x in color_nums and (z - x) in color_nums:
                return False
        return True

    def _get_obs(self):
        obs = np.zeros((200, 2), dtype=np.int32)
        for i, (num, color) in enumerate(self.pairs):
            obs[i] = [num, color]
        return obs

    def render(self):
        print(f"z={self.z-1}, Sequence: {self.pairs}")