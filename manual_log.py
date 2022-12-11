#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys
from skimage import io
import os
import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
from Howfast import Log


parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
parser.add_argument("--state", default=0)
parser.add_argument("--map-name", default="loop_only_duckies")
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

log=Log() # contains the transition and angle state to save.

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
        
# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    #MODIFY arrays as you wish
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    button=np.array([0,0,0,0,0]) # registers the button
    wheel_distance = 0.102
    min_rad = 0.08
    
    IMAGE_PATH = "DATA/" #where to save the pictures. MODIFY 
     # Where to save actions and properties: 
    BASE_PATH3 = "DATA/"

    action = np.array([0.0, 0.0]) 

    if key_handler[key.UP]:
        button[0]=1
        action += np.array([0.44, 0.0])  
    #select the transition based on the numerical buttons        
    if key_handler[key._1]:
        log.setTrans(1)
    if key_handler[key._2]:
        log.setTrans(2)
    if key_handler[key._3]:
        log.setTrans(3)
    if key_handler[key._0]:
        log.setTrans(0)
    #decrease the velocity
    if key_handler[key.S]:
        temp=log.getTran()-1
        log.setTrans(temp)
        
    
    if key_handler[key.DOWN]:
        action -= np.array([0.22, 0])
        button[1]=1
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
        button[2]=1
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
        button[3]=1
        
    if key_handler[key.SPACE]: #start logging
        if args.state==0:
            args.state=1;    
            print("LOGON")
            
        elif args.state==1: #stop logging
            args.state=0
            print("LOGOFF")
            
    if key_handler[key.LSHIFT]:
        action *= 1.5
        
        
    tran = log.getTran()+1 
   #Slow down or increase speed: 
    if tran== 1:
        action[0]*=0.5
    if tran==2:
        action[0]*=0.9
    if tran==3:
        action[0]*=1.2
    if tran==4 :
        action[0]*=1.5
        
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

    
    img=env.render('rgb_array')    
   
    env.render()
   
    name="test_"+str(env.unwrapped.step_count)+".jpeg"
    name2="test_"+str(env.unwrapped.step_count)+".txt"
    
    #save as many local data as possible for future improvments:
    if args.state==1:
        prop=np.array([action[0],action[1], env.unwrapped.cur_angle*360/3.140,  np.arctan(v1/v2)*360/3.14,log.getTran(), reward])
        io.imsave(os.path.join(IMAGE_PATH, name), img)
        print("SAVE")
        np.savetxt(os.path.join(BASE_PATH3, name2), prop)



pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
