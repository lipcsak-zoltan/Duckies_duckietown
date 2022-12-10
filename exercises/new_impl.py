import gym 
from stable_baselines3 import DQN
from stable_baselines3 import PPO, HerReplayBuffer
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
import os

import pyglet
from pyglet.window import key

import time
import sys

import math
import numpy as np
from gym_duckietown.envs import DuckietownEnv

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import cv2
import time
import tensorflow as tf


# class EnvWrapper(DuckietownEnv):    # The wrapper encapsulates the Duckietown env
#     def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
#         super().__init__(gain, trim, radius, k, limit, **kwargs)#, camera_width=80, camera_height=60)
#         self.observation_space = gym.spaces.Box(
#             low=0, high=255, shape=(80, 60, 3), dtype=np.uint8
#         )

#     def step(self, action):
#         obs, reward, done, info = super().step(action)   # calls the gym env methods
#         #obs = self._blur(obs)                             # applies your specific treatment
#         obs = cv2.resize(obs, (80, 60))

#         return obs, reward, done, info

# The main API methods that users of this class need to know are:

#         step
#         reset
#         render
#         close
#         seed

#     And set the following attributes:

#         action_space: The Space object corresponding to valid actions
#         observation_space: The Space object corresponding to valid observations
#         reward_range: A tuple corresponding to the min and max possible rewards

class SelfEnv(gym.Env):    # The wrapper encapsulates the Duckietown env
    def __init__(self, log_path, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        self.duckie_env = DuckietownEnv(gain, trim, radius, k, limit, **kwargs)#, camera_width=80, camera_height=60)

        #self.action_space = self.duckie_env.action_space
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(40, 80, 4), dtype=np.uint8
        )
        self.reward_range = self.duckie_env.reward_range

        self.file_writer = tf.summary.create_file_writer(log_path)
        self.step_idx = 0
        self.full_step_idx = 0
        self.acc_reward = 0
        self.iter_idx = 0

    def step(self, action):
        if action == 0:
            new_action = np.array([0.04, 0.4])
        elif action == 1:
            new_action = np.array([0.4, 0.04])
        elif action == 2:
            new_action = np.array([0.03, 0.3])
        obs, reward, done, info = self.duckie_env.step(new_action)   # calls the gym env methods
        #obs = self._blur(obs)                             # applies your specific treatment
        obs = self.preprocess_obs(obs)
        self.acc_reward += reward
        with self.file_writer.as_default():
            tf.summary.scalar('reward', reward, step=self.full_step_idx)
            if done:
                tf.summary.scalar('reward/acc_reward', self.acc_reward, step=self.iter_idx)
                tf.summary.scalar('reward/episode_len', self.step_idx, step=self.iter_idx)
        self.full_step_idx += 1
        self.step_idx += 1

        
        return obs, reward, done, info

    def reset(self, segment=False):
        obs = self.duckie_env.reset(segment)
        obs = self.preprocess_obs(obs)
        self.step_idx = 0
        self.acc_reward = 0
        self.iter_idx += 1
        return obs

    def render(self, mode = "human", close = False, segment = False):
        return self.duckie_env.render(mode, close, segment)
    
    def close(self):
        return self.duckie_env.close()

    def seed(self, seed=None):
        return self.duckie_env.seed(seed)
    
    def preprocess_obs(self, obs):
        obs = cv2.resize(obs, (80, 60))
        obs = obs[20:,:,:]

        hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

        red_mask = cv2.inRange(hsv, (116, 35, 0), (179, 255, 255))
        yellow_mask = cv2.inRange(hsv, (21, 61, 151), (75, 255, 255))
        white_mask = cv2.inRange(hsv, (0, 0, 137), (150, 36, 255))
        black_mask = cv2.inRange(hsv, (0, 0, 0), (179, 120, 95))

        obs = np.stack([red_mask, yellow_mask, white_mask, black_mask], axis=2)
        return obs

# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten()
#             # nn.Linear(128),
#             # nn.ReLU(),
#             # nn.Linear(3),
#             # nn.Softmax()
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

now = time.localtime()
subdir = time.strftime("%d-%b-%Y_%H.%M.%S", now)
log_path = os.path.join('Training', 'Logs', subdir)
orig_log_path = os.path.join('Training', 'Logs')

#new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
#env = DuckietownEnv(map_name = "udem1", domain_rand = False, draw_bbox = False)
env = SelfEnv(log_path, map_name = "loop_empty", domain_rand = False, draw_bbox = False)


#obs = env.reset()
#env.render()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=orig_log_path)
#model.set_logger(new_logger)

model.learn(1000000)


# env = DuckietownEnv(map_name = "zigzag_dists", domain_rand = False, draw_bbox = False) 
# model.set_env(env)
# model.learn(total_timesteps=100)

# env = DuckietownEnv(map_name = "small_loop_cw", domain_rand = False, draw_bbox = False) 
# model.set_env(env)
# model.learn(total_timesteps=100)

# env = DuckietownEnv(map_name = "zigzag_dists", domain_rand = False, draw_bbox = False) 
#  model.set_env(env)

model.save("first_model")

#valuate_policy(model, env, n_eval_episodes=10, render=True)

env.close()
