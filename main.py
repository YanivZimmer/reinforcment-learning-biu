# imports
print("start imports")
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

os.environ['LANG'] = 'en_US'
import time
from math import floor
import logging
from tensorflow.keras.layers import Dense
from gym.wrappers.time_limit import TimeLimit
from tensorflow.keras import backend as Keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# logger
ticks = str(floor(time.time()))
logging.basicConfig(filename='bipedal_walker_time_{0}.log'.format(ticks), level=logging.INFO)
# check gpu
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    print(device)


# video
def wrap_env(env):
    env = Monitor(env, './video', force=True)
    # env = TimeLimit(env, max_episode_steps=1000)
    return env


# make env
env = gym.make('BipedalWalker-v3').env
env = wrap_env(env)
print(env.spec)
state_size = env.observation_space


def change_reward(reward):
    if reward == -100:
        return -10
    return reward


def train_step(agent, envir):
    observation = envir.reset()
    done = False
    score = 0
    steps_counter = 0
    while not done:
        action_idx,action = agent.choose_action(observation)
        next_observation, reward, done, info = envir.step(action)
        steps_counter += 1
        reward = change_reward(reward)
        score += reward
        # memory
        agent.save_transition(observation, action_idx, action, reward,
                              next_observation, done)
        agent.learn_batch()
        observation = next_observation
    return score
    # ...


def train_loop(agent, episodes, envir):
    score_history = []
    max_score = float('-inf')
    logging.info("start train loop for agent {0}".format(agent.get_name()))

    for episode_idx in range(episodes):
        score = train_step(agent, envir)
        agent.calculate_epsilon()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        max_score = max(max_score, score)
        logging.info(
            'episode {0} score {1} avg_score {2} epsilon {3}'.format(episode_idx, score, avg_score, agent.epsilon))
    logging.info("Training is complete")

#mave env disc
"""## Make env discrete"""
def create_discrete_space(min_vals, max_vals, intervals=11):
    # action_space_matrix = np.zeros(shape=(len(min_vals), intervals))
    action_space_matrix = [[] * intervals] * len(min_vals)
    for min, max, i in zip(min_vals, max_vals, range(len(min_vals))):
        jump = (max - min) / (intervals - 1)
        action_space_matrix[i] = [min + jump * j for j in range(intervals)]
    return action_space_matrix

def list_permutations(arr, candidate=[]):
    if len(arr) == 0:
        return [candidate]
    all_res = []
    for item in arr[0]:
        new_candidate = candidate + [item]
        res = list_permutations(arr[1:], new_candidate)
        if len(res) != 0:
            all_res.extend(res)
    return all_res

def action_index_to_coordinates(action_idx):
    return all_perm[action_idx]

"""### Action space"""
# Get action min and max values
action_space_raw = env.action_space
action_min_vals = action_space_raw.low
action_max_vals = action_space_raw.high

# Create action space matrix
action_space_options_matrix = create_discrete_space(action_min_vals, action_max_vals, 5)
action_space_vectors = list_permutations(action_space_options_matrix)


"""### Observation space"""
#TODO- get min\max for each courdinate
observation_space = env.observation_space
observation_min_vals = observation_space.low
observation_max_vals = observation_space.high
# random sample 100000 and take min and max
max_bin = 0
min_bin = 0
for i in range(10000):
    sample = observation_space.sample()
    cur_min = np.min(sample)
    cur_max = np.max(sample)
    min_bin = min(cur_min, min_bin)
    max_bin = max(cur_max, max_bin)

# round up and down
max_bin = np.round(max_bin) + 1
min_bin = np.round(min_bin) - 1
state_min_vals = [
    0, min_bin, -1, -1,
    min_bin, min_bin, min_bin, min_bin,
    0, min_bin, min_bin, min_bin,
    min_bin, 0, min_bin, min_bin,
    min_bin, min_bin, min_bin, min_bin,
    min_bin, min_bin, min_bin, min_bin
]
state_max_vals = [
    2 * np.pi, max_bin, 1, 1,
    max_bin, max_bin, max_bin, max_bin,
    1, max_bin, max_bin, max_bin,
    max_bin, 1, max_bin, max_bin,
    max_bin, max_bin, max_bin, max_bin,
    max_bin, max_bin, max_bin, max_bin
]
state_matrix = create_discrete_space(state_min_vals, state_max_vals, 24)
state_matrix[8] = [0, 1]
state_matrix[13] = [0, 1]

def map_state_to_bins(state, state_mat):
    res = []
    for val, bins in zip(state, state_mat):
        matched_bin = np.digitize(val, bins) - 1
        if matched_bin == -1:
            matched_bin = 0
        res.append(bins[matched_bin])
    return np.asarray(res, dtype=np.float32)


class ReplayMemory:
    def __init__(self, max_size, state_dim, discreteStates, discreteActions):
        self.memory_size = max_size
        self.memory_counter = 0
        # discrete env space
        states_dtype = np.int32 if discreteStates else np.float32
        actions_dtype = np.int32 if discreteActions else np.float32
        self.states = np.zeros((self.memory_size, state_dim), dtype=states_dtype)
        self.states_next = np.zeros((self.memory_size, state_dim), dtype=states_dtype)
        # action space discrete
        self.actions_idx = np.zeros(self.memory_size,dtype=int)
        self.actions = np.zeros((self.memory_size,4),dtype=float)
        # rewards are floatType
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        # boolean but will be represeted as int
        self.is_terminal = np.zeros(self.memory_size, dtype=np.int32)

    def save_transition(self, state, action_idx,action, reward, state_next, is_terminal):
        idx = self.memory_counter % self.memory_size
        self.states[idx] = state
        self.states_next[idx] = state_next
        self.actions_idx[idx]=action_idx
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.is_terminal[idx] = is_terminal
        self.memory_counter += 1

    def sample_batch(self, batch_size):
        max_available = min(self.memory_size, self.memory_counter)
        batch = np.random.choice(max_available, batch_size, replace=False)
        states = self.states[batch]
        states_next = self.states_next[batch]
        rewards = self.rewards[batch]
        actions_idx = self.actions_idx[batch]
        actions = self.actions[batch]
        is_terminal = self.is_terminal[batch]

        return states, actions_idx, actions, rewards, states_next, is_terminal
class dummy_agent:
    def __init__(self):
        self.epsilon = 1

    def get_name(self):
        return "Dummy agent"

    def calculate_epsilon(self):
        logging.debug("dummy agent calculate_epsilon")
        pass

    def choose_action(self, observation):
        action = np.random.rand(4)
        logging.debug("dummy agent choose_action")
        return 0,action

    def save_transition(self, observation, action, reward,
                        next_observation, done):
        transition = "action {} reward {} next_observation {} done {})" \
            .format(action, reward, next_observation, done)
        logging.debug("dummy agent save transition: {}".format(transition))

    def learn_batch(self):
        logging.debug("dummy agent learn batch")


class ActorCritic:
    def __init__(self, memory, batch_size,input_len,action_space, epsilon_dec, epsilon_end
                 ,alpha,beta,gamma,clip_value,layer1_size,layer2_size):
        self.memory = memory
        self.batch_size = batch_size
        self.input_length=input_len
        self.epsilon = 1
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.num_actions = len(action_space)
        self.actor, self.critic, self.policy = self.create_network(layer1_size=layer1_size,layer2_size=layer2_size,num_actions=self.num_actions,alpha=alpha,beta=beta)
        # Discount factor
        self.gamma = gamma
        self.clip_value = clip_value
        self.episode = 0
        self.input_length = state_size
        self.action_space = action_space
        self.action_space_indices = [i for i in range(self.num_actions)]

    def create_network(self,layer1_size,layer2_size,num_actions,alpha,beta):
        input = Input(shape=(self.input_length,))
        # For loss calculation
        delta = Input(shape=[1])
        first_layer = Dense(layer1_size, activation='relu')(input)
        bn1 = tf.keras.layers.BatchNormalization()(first_layer)
        second_layer = Dense(layer2_size, activation='relu')(bn1)
        bn2 = tf.keras.layers.BatchNormalization()(second_layer)
        probabilities = Dense(num_actions,
                              activation='softmax')(bn2)
        values = Dense(1, activation='linear')(bn2)

        def custom_loss(actual, prediction):
            # We clip values so we dont get 0 or 1 values
            out = Keras.clip(prediction,1e-9, 1 - 1e-9)
            # Calculate log-likelihood
            likelihood = actual * tf.math.log(out)

            loss = tf.reduce_sum(-likelihood * delta)
            return loss

        actor = Model([input, delta], [probabilities])
        actor.compile(optimizer=Adam(lr=alpha), loss=custom_loss)
        critic = Model([input], [values])
        critic.compile(optimizer=Adam(lr=beta), loss='mean_squared_error')
        policy = Model([input], [probabilities])
        return actor, critic, policy

    def get_name(self):
        return "Actor Critic agent"

    def calculate_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def choose_action_index(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space_indices)

        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        # probabilities = Keras.clip(probabilities, self.clip_value, 1 - self.clip_value)
        probabilities = np.where(probabilities > (1 - self.clip_value), (1 - self.clip_value), probabilities)
        probabilities = np.where(probabilities < self.clip_value, self.clip_value, probabilities)
        probabilities = np.where(np.isnan(probabilities), self.clip_value, probabilities)

        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        return np.random.choice(self.action_space_indices, p=probabilities)

    def choose_action(self, observation):
        action_index = self.choose_action_index(observation)
        action = self.action_space[action_index]
        return action_index,action

    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)

    def learn_batch(self):
        if self.memory.memory_counter < self.batch_size:
            print("Return::batch size bigger than memory counter :batch size={0} memory_counter={1}"\
                  .format(self.batch_size, self.memory.memory_counter))
            return

        states, actions_idx, actions, rewards, next_states, is_terminal = self.memory.sample_batch(self.batch_size)

        critic_next_value = self.critic.predict(next_states)
        critic_value = self.critic.predict(states)
        non_terminal = np.where(is_terminal == 1, 0, 1)
        # 1 - int(done) = do not take next state into consideration if done
        target = rewards + self.gamma * np.max(critic_next_value) * non_terminal
        delta = target - np.concatenate(critic_value)
        self.critic.fit(states, target, verbose=0, batch_size=self.batch_size)
        self.actor.fit([states, delta], actions_idx, verbose=0, batch_size=self.batch_size)


print("start")
#ag = dummy_agent()
states_dim = len(state_matrix)
mem = ReplayMemory(5000, states_dim, True, True)
lr=0.0005
ag=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,action_space=action_space_vectors,epsilon_dec=0.001,epsilon_end=0.05,alpha=lr,
               beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=256,layer2_size=256)
train_loop(ag, 5, env)
