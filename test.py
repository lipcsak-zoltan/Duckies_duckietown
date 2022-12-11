#!/usr/bin/env python
# manual

import argparse
import sys
from model import veloCNN, angleCNN
import cv2
import torch 
import gym
import numpy as np
import pyglet
from pyglet.window import key
from torchvision import transforms 

from gym_duckietown.envs import DuckietownEnv

trafo = transforms.ToTensor()
parser = argparse.ArgumentParser()


parser.add_argument("--env-name", default="Duckietown-udem1-v0")
parser.add_argument("--state", default=0)
parser.add_argument("--map-name", default="loop_empty")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()


total_reward=0
if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

velmodel=(torch.load("velocity.pth", map_location='cpu')) #adjust for proper location

angmodel=(torch.load("angle.pth", map_location='cpu'))


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
        
# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):

    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    button=np.array([0,0,0,0])
    wheel_distance = 0.102
    min_rad = 0.08
    img=env.render('rgb_array') 
    hls = hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_lower = np.array([np.round(  0 / 2), np.round(0.5 * 255), np.round(0.00 * 255)])

    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
    white_mask = cv2.inRange(hls, white_lower, white_upper)


    yellow_lower = np.array([np.round( 40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
    yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    maski = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(img, img, mask=maski)
    ksize  = (7, 7)
    masked = cv2.blur(masked, ksize)
    output= masked[200:800, 20:500]
    output = cv2.resize(output,[100, 75])

    action = np.array([0.0, 0.0])
    
    data = trafo(output)
    data/=255
    data=data.unsqueeze(0)

    print(data.shape)
    velout = velmodel(data)
    angout = angmodel(data)
    button=velout.detach().cpu().numpy()
    angbut = angout.detach().cpu().numpy()

    print(button.shape)
    print(button[0][1])
    
    temp = 0
    state = 0
    
    for i in range(0,3):
        if temp < button[0][i]:
            temp= button[0][i]
            state= i+1
    
    print(state)
    action += np.array([0.25, 0.0])  

    
    if angbut[0][2] > 0:
        action += np.array([0, 1])
    if angbut[0][0] > 0:
        action -= np.array([0, 1]) 
        
    if state== 1:
        action[0]*=1
    if state==2:
        action[0]*=1.2
    if state==3:
        action[0]*=1.5
    if state==4 :
        action[0]*=2
         
    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled

        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2
 
    obs, reward, done, info = env.step(action) 
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward)) 
    env.render()     
    #np.savez(name,png)
   # io.imsave('screenshot.png', img)

    if done:
        env.reset()
        
pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()