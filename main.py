#imports
import matplotlib.pyplot as plt
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import os
import time
from math import floor
import logging
#logger
ticks = floor(time.time())
logging.basicConfig(filename='{0}.log'.format(ticks), encoding='utf-8', level=logging.DEBUG)
#check gpu
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    print(device)
#video
def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

#make env
env = gym.make('BipedalWalker-v3')
env = wrap_env(env)
state_size = env.observation_space

def train_step(agent,envir,max_steps=350,min_score=float('-inf')):
    observation = envir.reset()
    done = False
    score = 0
    steps_counter = 0
    # ...
    return 0, 0

def train_loop(agent,episodes,envir,max_steps=350,min_score=float('-inf')):
    score_history = []
    max_score = float('-inf')
    logging.info("start train loop for agent {0}".format(agent.name))

    for episode_idx in range(episodes):
        score,steps = train_step(agent,envir,max_steps,min_score)
        agent.calculate_epsilon()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        max_score = max(max_score, score)
        logging.info( 'episode {0} score {1} 100_moving_window_score {2} epsilon {3} actions_per_epoch {4}'.format(episode_idx,score,avg_score,agent.epsilon,steps))

