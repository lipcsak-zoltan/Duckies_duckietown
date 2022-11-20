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

We used 2 approach to this milestone
Markup: *Use existing RL policy as starting point 
        *Creating a policy from scratch
Based on further tests and experiments we are going to decide which one to use as a final policy.
