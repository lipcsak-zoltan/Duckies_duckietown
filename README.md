# Duckietown project by the Duckies
## Team Duckies  
Members: Lipcsák Zoltán István (XWSJPH), Pólya Mátyás Dániel (BBAGDV), Katona Axel Attila (TAUHO5)
## Topic
Autonomous driving in Duckietown enviroment  
## Project description 
Facilitation of autonomous driving in Duckietown with Reinforcement Learning methods.
## Subject
Deep Learning in Practice with Python and LUA (BMEVITMAV45)  

## Installation
Navigate to root of the project

`pip3 install -e .`
## Exercises
### Running manual_control.py
`python manual_control.py --env-name Duckietown`
### Creating a new map
We have created two new maps, named new_map.yaml and customMap.yaml, found in Duckies_duckietown/gym_duckietown/maps/

To use them, copy them to the folder containing the original maps (if given the --map-name parameter, the program logs where it searches for it, e.g. lib/python3.6/site-packages/duckietown_world/data/gd1/maps)

### Running the basic_control.py
`python basic_control.py --map-name new_map`

The run is recorded in the dl_basic.mp4 file

## Milestone 2

We used 2 approaches to this milestone:
* Use [AIDO](https://docs.duckietown.org/daffy/AIDO/out/embodied_rl.html)RL policy as starting point 
* Use [RL baseline 2](https://github.com/nicknochnack/ReinforcementLearningCourse/blob/main/Project%202%20-%20Self%20Driving.ipynb) policy 
<a/>
Based on further tests and experiments we are going to decide which one to use as a final policy.

## RL_AIDO

See requirements.txt for the required packages in [RL_AIDO](/RL_AIDO)  folder. 
Install them with 

`$ pip3.8 install -e .`


(It was tested with Python 3.8).

Start training with

`$ cd duckietown_rl`

`$ python3.8 -m scripts.train_cnn --seed 123`

We tried to train the network in Linux. Some graphs avaliable [here](https://wandb.ai/dodekaeder/test1/reports/episode-reward-22-11-20-21-25-30---VmlldzozMDA1MTUy?accessToken=bcy084vs1ah194odlrbtds38ire0aljs61d9h3x2h2svbcyd4buax16fjw0l2h79) and we also updated them to the "results" folder.

Based on the previous experiments, the training parameter needs to be chosen carefully. For now, we only trained the policy on CPU, so it was slower than we expected. Hopefully GPU training will be available soon to set the parameters ideally to obtain solid results.
