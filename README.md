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
The project was tested under python3.8, and CUDA11.7.
## Exercises
### Running manual_control.py
`python manual_control.py --env-name Duckietown`
## Start logging
* Run
`manual_test.py`
    * You can modify action[] entries for different keys
    * navigate in the map with the keyboard
* Press SPACE to start logging 
* Make sure Howfast.py is in the same directory as 
`manual_test.py`

## Preprocess data
* The raw images saved before needs to be pre-processed: run 
`preprocess.py`
with the appropriate file names you logged.


## Train on data 
* Run 
`train.py` 
in /Train/ directory. 
* Don't forget to sign in with yout wandb credits, and modify those in the script
* Also check every path for training data
## Test
* Run
`test.py` 
* See 
`testrun1.mp4` 
* See
`Unknown_test.mkv` to check how the prediction works on an unknow, complex map. 
* See Train/log folder for more log files, wandb accuracies, losses. Accuricies are NOT in percent, they are dimension free rletive quantities, i.e. 1.0 = 100%.
