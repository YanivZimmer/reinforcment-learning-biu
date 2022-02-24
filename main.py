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
from tensorflow.keras.layers import Dense
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

def check_episode_exceeded(score,min_score,steps,max_steps):
    if score<min_score:
        logging.debug("stoped episode due to score {} lower than {}".format(score,min_score))
    if steps>max_steps:
        logging.debug("stoped episode due to steps {} higher than {}".format(steps,max_steps))
        return True
    return False
def change_reward(reward):
    if reward==-100:
        return -10
    return reward
def train_step(agent,envir,max_steps=350,min_score=float('-inf')):
    observation = envir.reset()
    done = False
    score = 0
    steps_counter = 0
    while not done:
        action=agent.choose_action(observation)
        next_observation, reward, done, info = envir.step(action)
        steps_counter+=1
        if check_episode_exceeded(score,min_score,steps_counter,max_steps):
            done=True
        else:
            done=False
        reward=change_reward(reward)
        score+=reward
        #memory
        agent.save_transition(observation, action, reward,
                              next_observation, done)
        agent.learn_batch()
        observation = next_observation
    return score
    # ...

def train_loop(agent,episodes,envir,max_steps=350,min_score=float('-inf')):
    score_history = []
    max_score = float('-inf')
    logging.info("start train loop for agent {0}".format(agent.get_name()))

    for episode_idx in range(episodes):
        score = train_step(agent,envir,max_steps,min_score)
        agent.calculate_epsilon()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        max_score = max(max_score, score)
        logging.info( 'episode {0} score {1} 100_moving_window_score {2} epsilon {3}'.format(episode_idx,score,avg_score,agent.epsilon))
    logging.info("Training is complete")


class dummy_agent():
    def __init__(self):
        pass

    def get_name(self):
        return "Dummy agent"

    def calculate_epsilon(self):
        logging.debug("dummy agent calculate_epsilon")
        pass

    def choose_action(self, observation):
        action= [-1, -1, -1, -1]
        logging.debug("dummy agent choose_action")

    def save_transition(self, observation, action, reward,
                          next_observation, done):
        transition= "observation {} action {} reward {} next_observation {} done {})"\
            .format(observation, action, reward,next_observation, done)
        logging.debug("dummy agent save transition: {}".format(transition))

    def learn_batch(self):
        logging.debug("dummy agent learn batch")
