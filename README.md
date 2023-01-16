# Duckietown project by the Duckies
# UPDATE: VIDEO FOR THE EXAM on 17th Jan
## The video for the exam is available [here](https://bmeedu-my.sharepoint.com/:v:/g/personal/katonaa_edu_bme_hu/EY0erGUlVk5Cg3uGPV0VZ5YBVsuRln5eLpTaA6SPBDlR9w?e=vRC6q6)
## Team Duckies  
Members: Lipcsák Zoltán István (XWSJPH), Pólya Mátyás Dániel (BBAGDV), Katona Axel Attila (TAUHO5)
## Topic
Autonomous driving in Duckietown enviroment  

## Project description 
Facilitation of autonomous driving in Duckietown with Reinforcement Learning methods.
## Subject
Deep Learning in Practice with Python and LUA (BMEVITMAV45)  
## Reinforcement Learning Implemetation
For the Reinforcement Learning Implemetation see the RL_milestone3 branch
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
* A fraction of the training data can be found [here](https://bmeedu-my.sharepoint.com/:f:/g/personal/katonaa_edu_bme_hu/Eo-8rKw1fv9GjJUyLXWmdbMB44LxB9gA2NvWmQsKA9xkRA?e=4162CH)
## Test
* Run
`test.py` 
* See 
`testrun1.mp4` 
* See
`Unknown_test.mkv` to check how the prediction works on an unknow, complex map. 
* See Train/log folder for more log files, wandb accuracies, losses. Accuracies are NOT in percent, they are dimension free relative quantities, i.e. 1.0 = 100%.
