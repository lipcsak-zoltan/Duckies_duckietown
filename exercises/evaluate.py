import gym 
from stable_baselines3 import DQN, HerReplayBuffer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

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

class SelfEnv(gym.Env):    # The wrapper encapsulates the Duckietown env
    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        self.duckie_env = DuckietownEnv(gain, trim, radius, k, limit, **kwargs)#, camera_width=80, camera_height=60)

        #self.action_space = self.duckie_env.action_space
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(40, 80, 3), dtype=np.uint8
        )
        self.reward_range = self.duckie_env.reward_range

        self.step_idx = 0
        self.full_step_idx = 0
        self.acc_reward = 0
        self.iter_idx = 0
        #self.obs_seq = np.zeros((40, 80, 20))
    def step(self, action):
        if action == 0: # left
            new_action = np.array([0.0, 1])
        elif action == 1: # forward
            new_action = np.array([0.88, 0.44])
        elif action == 2: # right
            new_action = np.array([0, -1])
        obs, reward, done, info = self.duckie_env.step(new_action)   # calls the gym env methods
        #obs = self._blur(obs)                             # applies your specific treatment
        obs = self.preprocess_obs(obs)
        self.acc_reward += reward
        self.full_step_idx += 1
        self.step_idx += 1

        if done:
            reward = -30
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
        #black_mask = cv2.inRange(hsv, (0, 0, 0), (179, 120, 95))

        obs = np.stack([red_mask, yellow_mask, white_mask], axis=2)
        #self.obs_seq = np.concatenate((obs, self.obs_seq[:,:,:16]), axis = 2)
        return obs

    
env = SelfEnv(map_name = "loop_empty", domain_rand = False, draw_bbox = False)

model = DQN.load("/home/user/Desktop/duckies/Duckies_duckietown/exercises/Training/Logs/11-Dec-2022_13.09.12/rl_model_5000_steps.zip")
obs = env.reset()
cumreward = 0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        env.reset()
        print("cumreward = ", cumreward)
        cumreward = 0
        continue
    print(action, rewards, end= "\n")
    cumreward += rewards
    env.render()

env = SelfEnv(map_name = "udem1", domain_rand = False, draw_bbox = False)
obs = env.reset()
cumreward = 0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        env.reset()
        print("cumreward = ", cumreward)
        cumreward = 0
        continue
    print(action, rewards, end= "\n")
    cumreward += rewards
    env.render()

env = SelfEnv(map_name = "zigzag_dists", domain_rand = False, draw_bbox = False)
obs = env.reset()
cumreward = 0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        env.reset()
        print("cumreward = ", cumreward)
        cumreward = 0
        continue
    print(action, rewards, end= "\n")
    cumreward += rewards
    env.render() 
    
