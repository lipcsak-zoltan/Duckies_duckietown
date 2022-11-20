import gym 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
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



log_path = os.path.join('Training', 'Logs')

env = DuckietownEnv(map_name = "udem1", domain_rand = False, draw_bbox = False)

#obs = env.reset()
#env.render()

model = A2C("MlpPolicy", env, verbose=3, tensorboard_log=log_path)
model.learn(100,progress_bar=True)

env = DuckietownEnv(map_name = "zigzag_dists", domain_rand = False, draw_bbox = False) 
model.set_env(env)
model.learn(total_timesteps=100,progress_bar=True)

env = DuckietownEnv(map_name = "small_loop_cw", domain_rand = False, draw_bbox = False) 
model.set_env(env)
model.learn(total_timesteps=100,progress_bar=True)

env = DuckietownEnv(map_name = "zigzag_dists", domain_rand = False, draw_bbox = False) 
model.set_env(env)
evaluate_policy(model, env, n_eval_episodes=10, render=False)

env.close()
