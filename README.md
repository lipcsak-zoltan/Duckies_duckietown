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

# Milestione II version 2
This branch contains the stable-baseline3 version of the reinforcement learning task

Docker is used for the easier reusability
## Creating the environment
Navigate to the root of the project

`sudo docker build . \
       --file ./docker/standalone/Dockerfile \
       --no-cache=true \
       --network=host \
       --tag duckie`
       
`docker run -it duckie bash`

`Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:0 `

`pip3 install -e .`

`pip install  numpy==1.23`

`export PYTHONPATH="${PYTHONPATH}:'pwd'"`

## Starting the learning

Navigate to exercises

`python3 new_impl.py`
