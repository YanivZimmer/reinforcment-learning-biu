# imports
print("start imports")
from pyclbr import Function
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.spaces.discrete import Discrete
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import os
import copy
from enum import Enum

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
from datetime import datetime
import pylab

# Disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class ModelType(Enum):
    ACTOR_CRITIC = 1
    DQN = 2

# Run settings
OPENAI_ENV = 'LunarLanderContinuous-v2' # 'BipedalWalker-v3'
MODEL = ModelType.ACTOR_CRITIC
NUM_EPOCHS = 3000
MAKE_ACTION_DISCRETE = True
NUM_ACTION_BINS = [4, 4]
MAKE_STATE_DISCRETE = False
NUM_STATE_BINS = 5
MEMORY_SIZE = 5000
BATCH_SIZE = 64
LAYER1_SIZE = 64
LAYER2_SIZE = 32
LAYER1_ACTIVATION = 'tanh' #'relu'
LAYER2_ACTIVATION = 'tanh' #'relu'
EPSILON = 1
EPSILON_DEC_RATE = 0.001
EPSILON_MIN = 0.01
GAMMA = 0.99
LEARNING_RATE = 0.00025
lr_low1 = 0.00002 #0.35 * LEARNING_RATE
lr_low2 = 0.00002 #0.25 * LEARNING_RATE
FIT_EPOCHS = 10
CLIP_VALUE = 1e-9
MODIFY_REWARD = False
MODIFIED_REWARD = -10 if MODIFY_REWARD else -100

# Do not explore when using ActorCritic
if MODEL is not ModelType.ACTOR_CRITIC:
    EPSILON = 1
    EPSILON_DEC_RATE = 0.001
    EPSILON_MIN = 0.01

# logger
date_str = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,handlers=[
        logging.FileHandler(f'logs/{OPENAI_ENV}_time_{date_str}.log'),
        logging.StreamHandler()
    ])

# check gpu
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    print(device)


# video
# def show_video():
#   mp4list = glob.glob('video/*.mp4')
#   if len(mp4list) > 0:
#     mp4 = mp4list[0]
#     video = io.open(mp4, 'r+b').read()
#     encoded = base64.b64encode(video)
#     ipythondisplay.display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#   else:
#     print("Could not find video")
#
def wrap_env(env):
    env = Monitor(env, './video', force=True)
    #env = TimeLimit(env, max_episode_steps=1000)
    return env

def PlotModel(episode, scores, averages):
    episodes = [i for i in range(0, episode + 1)]

    if str(episode)[-1:] == "0":# much faster than episode % 100
        pylab.plot(episodes, scores, 'b')
        pylab.plot(episodes, averages, 'r')
        pylab.title(f'{OPENAI_ENV} PPO training cycle\n\
                        lr actor = {lr_low1}, lr critic = {lr_low2}, epsilon = ({EPSILON},{EPSILON_DEC_RATE},{EPSILON_MIN})\n\
                        layer1 = {LAYER1_SIZE}, layer2 = {LAYER2_SIZE}, action bins = {NUM_ACTION_BINS}, batch = {BATCH_SIZE}\n\
                        reward = {MODIFIED_REWARD}, fit epochs = {FIT_EPOCHS}', fontsize=8)
        pylab.ylabel('Score', fontsize=10)
        pylab.xlabel('Steps', fontsize=10)
        try:
            pylab.grid(True)
            pylab.savefig(f'plots/{OPENAI_ENV}-{date_str}.png')
        except OSError:
            pass

# make env
env = gym.make(OPENAI_ENV).env
#env = TimeLimit(env, max_episode_steps=1000)
#env = wrap_env(env)
print(env.spec)

"""### Action space"""
# Get action min and max values
action_space = env.action_space
action_min_vals = action_space.low
action_max_vals = action_space.high

actions_dim = len(action_min_vals)

"""### Observation space"""
observation_space = env.observation_space
observation_min_vals = observation_space.low
observation_max_vals = observation_space.high

states_dim = len(observation_min_vals)

# class Discretizationer:
#     def __init__(self,bins_num,min,max):
#         self.bins_num = bins_num
#         self.min=min
#         self.max=max
#         self.interval_size = (max - min) / bins_num
#     def val_to_bin(self,val):
#         if(val==self.max):
#             return self.bins_num-1
#         bin_of = np.floor((val-self.min)/self.interval_size)
#         return bin_of
#     def vec_val_to_bin(self,vec):
#         bin_vec = np.zeros(len(vec))
#         for i in range(len(vec)):
#             bin_vec[i] = self.val_to_bin(vec[i])
#         return bin_vec
#     def bin_to_val(self,bin):
#         #get midian value of bin
#         val=(bin-0.5)*self.interval_size
#         return val
#     def vec_bin_to_vals(self,bin_vec):
#         val_vec = np.zeros(len(bin_vec))
#         for i in range(len(bin_vec)):
#             val_vec[i] = self.bin_to_val(bin_vec[i])
#         return val_vec
#     def vec_val_to_idx(self,vec):
#         sum=0
#         bin_vec=self.vec_val_to_bin(vec)
#         length=len(bin_vec)
#         for i in range(length):
#             j=np.power(self.bins_num,i)
#             sum += (bin_vec[i] * j)
#         return sum
#     def idx_to_val_vec(self,idx,vec_size):
#         bin_vec=np.zeros(vec_size)
#         for i in range(vec_size):
#             j=np.power(self.bins_num,vec_size-i-1)
#             bin_of=np.floor(idx/j)
#             idx-=(bin_of*j)
#             bin_vec[vec_size-i-1]=bin_of
#         #print(bin_vec)
#         return self.vec_bin_to_vals(bin_vec)

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """
    This wrapper converts an observation into bins.
    """
    def __init__(self, env, num_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(self.observation_space, spaces.Box)

        self.low = self.observation_space.low if low is None else low
        self.high = self.observation_space.high if high is None else high
        self.num_bins = num_bins
        self.state_matrix = self._create_discrete_space()
    
    # def _create_discrete_space(self):
    #     space_matrix = [[] * self.num_bins] * len(self.low)
    #     for min, max, i in zip(self.low, self.high, range(len(self.low))):
    #         jump = (max - min) / (self.num_bins - 1)
    #         space_matrix[i] = [min + jump * j for j in range(self.num_bins)]
    #     return space_matrix
    
    def _create_discrete_space(self):
        return [np.linspace(low, high, num=self.num_bins)
                for low, high in zip(self.low, self.high)]
    
    # def _map_state_to_bins(self, observation):
    #     res = []
    #     for val, bins in zip(observation, self.state_matrix):
    #         matched_bin = np.digitize(val, bins) - 1
    #         if matched_bin == -1:
    #             matched_bin = 0
    #         res.append(bins[matched_bin])
    #     return np.asarray(res, dtype=np.float32)

    def observation(self, observation):
        # print(f'observation: {observation}')
        binned_observation = [bins[np.digitize([x], bins)[0]]
                  for x, bins in zip(observation, self.state_matrix)]
        # print(f'binned_observation: {binned_observation}')
        # print(f'state_matrix: {self.state_matrix}')
        return binned_observation

class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    This wrapper converts an action into bins.
    """
    def __init__(self, env, num_bins=[10, 10], low=None, high=None):
        super().__init__(env)
        assert isinstance(self.observation_space, spaces.Box)

        self.low = self.observation_space.low if low is None else low
        self.high = self.observation_space.high if high is None else high
        self.num_bins = num_bins
        self.action_matrix = self._create_discrete_space()
        self.all_perms = self._list_permutations(self.action_matrix)
    
    def _create_discrete_space(self):
        return [np.linspace(low, high, num=num_bin)
                for low, high, num_bin in zip(self.low, self.high, self.num_bins)]
    
    def _list_permutations(self, arr, candidate = []):
        if len(arr) == 0:
            return [candidate]
        
        all_res = []
        for item in arr[0]:
            new_candidate = candidate + [item]
            res = self._list_permutations(arr[1:], new_candidate)
            if len(res) != 0:
                all_res.extend(res)
        
        return all_res

    def index_to_action(self, idx):
        # print(idx)
        # print(len(self.all_perms))
        return self.all_perms[idx]

    def action(self, action):
        # print(f'action: {action}')
        binned_action = []
        for x, bins in zip(action, self.action_matrix):
            mapped_idx = np.digitize([x], bins)[0]
            if mapped_idx == len(bins):
                mapped_idx -= 1
            binned_action.append(bins[mapped_idx])
        # binned_action = [bins[np.digitize([x], bins)[0]]
        #           for x, bins in zip(action, self.action_matrix)]
        # print(f'binned_action: {binned_action}')
        # print(f'state_matrix: {self.state_matrix}')
        return binned_action
    
    def reverse_action(self, action):
        return super().reverse_action(action)

def get_bounds(space):
    # random sample 100000 and take min and max
    min_bins = np.zeros(shape=(len(observation_max_vals)))
    max_bins = np.zeros(shape=(len(observation_min_vals)))

    for _ in range(100000):
        sample = space.sample()
        min_bins = np.minimum(min_bins, sample)
        max_bins = np.maximum(max_bins, sample)

    # round up and down
    max_bins = np.round(max_bins)+1
    min_bins = np.round(min_bins)-1

    return min_bins, max_bins

observation_min_bins, observation_max_bins = get_bounds(observation_space)
print(f'observation max: {observation_max_bins}, observation min: {observation_min_bins}')

# action_min_bins, action_max_bins = get_bounds(action_space)
print(f'action max: {action_max_vals}, action min: {action_min_vals}')

if MAKE_STATE_DISCRETE:
    env = DiscretizedObservationWrapper(env, NUM_STATE_BINS, observation_min_bins, observation_max_bins)

if MAKE_ACTION_DISCRETE:
    env = DiscretizedActionWrapper(env, NUM_ACTION_BINS, action_min_vals, action_max_vals)

# one hot
# def to_one_hot(indeces, depth):
#     return tf.one_hot(indeces, depth)

def change_reward(reward):
    if MODIFY_REWARD and reward == -100:
        logging.info(f'Changed reward to {MODIFIED_REWARD}')
        return MODIFIED_REWARD
    return reward
    # if reward>0:
    #     reward=1.5*reward
    # return reward


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
    average_history = []
    max_score = float('-inf')
    max_average = float('-inf')
    logging.info("start train loop for agent {0}".format(agent.get_name()))

    for episode_idx in range(episodes):
        score = train_step(agent, envir)
        agent.calculate_epsilon()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        average_history.append(avg_score)

        if average_history[-1] > max_average:
            logging.info(f'Saving weights')
            agent.save()
        
        max_score = max(max_score, score)
        max_average = max(max_average, avg_score)

        logging.info(
            'episode {0} score {1} avg_score {2} max_score {3} epsilon {4}'\
                .format(episode_idx, score, avg_score, max_score, agent.epsilon))
        PlotModel(episode_idx, score_history, average_history)
    
    logging.info("Training is complete")

class ReplayMemory:
    def __init__(self, max_size, actions_dim, state_dim):
        self.memory_size = max_size
        self.memory_counter = 0

        self.actions_dim = np.prod(NUM_ACTION_BINS)
        
        self.states = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        self.states_next = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        # action space discrete
        self.actions_idx = np.zeros((self.memory_size, self.actions_dim),dtype=int)
        self.actions = np.zeros((self.memory_size, actions_dim),dtype=float)
        # rewards are floatType
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        # boolean but will be represeted as int
        self.is_terminal = np.zeros(self.memory_size, dtype=np.int32)

    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        idx = self.memory_counter % self.memory_size
        self.states[idx] = state
        self.states_next[idx] = state_next
        self.actions_idx[idx, action_idx] = 1.0
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
# class dummy_agent:
#     def __init__(self):
#         self.epsilon = 1

#     def get_name(self):
#         return "Dummy agent"

#     def calculate_epsilon(self):
#         logging.debug("dummy agent calculate_epsilon")
#         pass

#     def choose_action(self, observation):
#         action = np.random.rand(4)
#         logging.debug("dummy agent choose_action")
#         return 0,action

#     def save_transition(self, observation, action, reward,
#                         next_observation, done):
#         transition = "action {} reward {} next_observation {} done {})" \
#             .format(action, reward, next_observation, done)
#         logging.debug("dummy agent save transition: {}".format(transition))

#     def learn_batch(self):
#         logging.debug("dummy agent learn batch")


class ActorCritic:
    def __init__(self, memory, batch_size, input_len, actions_dim, index_to_action: Function, epsilon, epsilon_dec, epsilon_end,
                    alpha, beta, gamma, clip_value, layer1_size, layer2_size):
        self.memory = memory
        self.batch_size = batch_size
        self.input_length=input_len
        self.actions_dim = actions_dim
        self.index_to_action = index_to_action
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        # Discount factor
        self.gamma = gamma
        self.clip_value = clip_value
        self.episode = 0
        self.num_actions = np.prod(NUM_ACTION_BINS) #np.power(NUM_ACTION_BINS, actions_dim)
        self.actor, self.critic, self.policy = self.create_network(layer1_size = layer1_size, layer2_size = layer2_size,
                                                                    num_actions = self.num_actions, alpha = alpha, beta = beta)

    def critic_ppo_loss(self,values):
        def loss(y_true,y_pred):
            loss_clip = 0.2
            clip_val_loss = values + Keras.clip(y_pred-values,-loss_clip,loss_clip)
            v_loss1 = (y_true - clip_val_loss) ** 2
            v_loss2 = (y_true-y_pred) ** 2
            val_loss = 0.5 *Keras.mean(Keras.maximum(v_loss1,v_loss2))
            return val_loss
        return loss

    def actor_ppo_loss(self,y_true,y_pred):
        advantages = y_true[:,:1]
        prediction_picks =  y_true[:, 1:1+self.num_actions]
        actions = y_true[:, 1+self.num_actions:]
        #constants
        entropy_loss = 0.001
        loss_clip = 0.2
        clip_tresh=1e-10
        #prob
        prob = actions * y_pred
        old_prob = actions * prediction_picks
        prob = Keras.clip(prob,clip_tresh,1.0)
        old_prob = Keras.clip(old_prob,clip_tresh,1.0)

        ratio = Keras.exp(Keras.log(prob)-Keras.log(old_prob))
        p1 = ratio*advantages
        p2 = Keras.clip(ratio,min_value=1-loss_clip,max_value=1+loss_clip) * advantages

        actor_loss = -Keras.mean(Keras.minimum(p1,p2))
        entropy = -(y_pred * Keras.log(y_pred+clip_tresh))
        entropy = entropy_loss * Keras.mean(entropy)

        total_loss = actor_loss - entropy
        return total_loss

    def create_network(self,layer1_size,layer2_size,num_actions,alpha,beta):
        input = Input(shape=(self.input_length,))
        # For loss calculation
        delta = Input(shape=[1])
        #first_layer = Dense(layer1_size, activation='relu')(input)
        first_layer=Dense(layer1_size, activation=LAYER1_ACTIVATION, kernel_initializer= \
            tf.random_normal_initializer(stddev=0.01))(input)

        bn1 = tf.keras.layers.BatchNormalization()(first_layer)
        #second_layer = Dense(layer2_size, activation='relu')(bn1)
        second_layer=Dense(layer2_size, activation=LAYER2_ACTIVATION, kernel_initializer= \
            tf.random_normal_initializer(stddev=0.01))(bn1)
        bn2 = tf.keras.layers.BatchNormalization()(second_layer)
        probabilities = Dense(num_actions,
                              activation='softmax')(bn2)
        values = Dense(1, activation='linear')(bn2)

        def custom_loss(actual, prediction):
            # We clip values so we dont get 0 or 1 values
            out = Keras.clip(prediction,1e-4, 1 - 1e-4)
            # Calculate log-likelihood
            likelihood = actual * tf.math.log(out)
            #likelihood = tf.math.log(prediction)
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
    
    def _get_actor_weight_file_name(self):
        return f'weights/{OPENAI_ENV}-actor-{date_str}'
    
    def _get_critic_weight_file_name(self):
        return f'weights/{OPENAI_ENV}-critic-{date_str}'

    def calculate_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def choose_action_index(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)

        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        #probabilities = Keras.clip(probabilities, self.clip_value, 1 - self.clip_value)
        #probabilities = np.where(probabilities > (1 - self.clip_value), (1 - self.clip_value), probabilities)
        #probabilities = np.where(probabilities < self.clip_value, self.clip_value, probabilities)
        #probabilities = np.where(np.isnan(probabilities), self.clip_value, probabilities)

        # Normalize probabilities
        #probabilities = probabilities / Keras.sum(probabilities)
        #tf.random.categorical(probabilities,self.action_space_indices)
        #the_np = probabilities.eval()
        #tf.make_tensor_proto(probabilities)
        return np.random.choice(range(0, self.num_actions), p=probabilities)

    def choose_action(self, observation):
        action_index = self.choose_action_index(observation)
        # action = self.action_discretizationer.idx_to_val_vec(action_index, self.actions_dim)
        action = self.index_to_action(action_index)
        return action_index, action

    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)
    
    def save(self):
        self.actor.save_weights(self._get_actor_weight_file_name())
        self.critic.save_weights(self._get_critic_weight_file_name())
    
    def load(self):
        self.actor.load_weights(self._get_actor_weight_file_name())
        self.critic.load_weights(self._get_critic_weight_file_name())

    def learn_batch(self):
        if self.memory.memory_counter < self.batch_size:
            print("Return::batch size bigger than memory counter :batch size={0} memory_counter={1}"\
                  .format(self.batch_size, self.memory.memory_counter))
            return

        states, actions_idx, actions, rewards, next_states, is_terminal = self.memory.sample_batch(self.batch_size)
        critic_next_value = self.critic.predict(next_states)[:, 0]
        critic_value = self.critic.predict(states)[:, 0]
        non_terminal = np.where(is_terminal == 1, 0, 1)
        # print(f'rewards: {rewards}')
        # print(f'critic_value: {critic_value}')
        # print(f'critic_next_value: {critic_next_value}')
        # print(f'non_terminal: {non_terminal}')
        # 1 - int(done) = do not take next state into consideration if done
        target = rewards + self.gamma * critic_next_value * non_terminal
        # print(f'target: {target}')
        delta = target - critic_value
        # print(f'delta: {delta}')
        self.critic.fit(states, target, verbose=0, epochs=FIT_EPOCHS, batch_size=self.batch_size)
        self.actor.fit([states, delta], actions_idx, epochs=FIT_EPOCHS, verbose=0, batch_size=self.batch_size)

class AgentDDQN():
    def __init__(self, lr, gemma, action_space, batch_size, states_dim,Memory,
               epsilon, epsilon_dec=1e-3, epsilon_end=0.01,
                fname='dqn_model.h5'):
      self.model_file = fname
      self.gemma = gemma
      self.action_space = action_space
      self.epsilon = epsilon
      self.epsilon_dec = epsilon_dec
      self.epsilon_end = epsilon_end
      self.batch_size = batch_size
      self.lr=lr
      self.memory = Memory
      self.q_eval1 = self._build_dqn(lr,len(action_space),states_dim)
      self.q_eval2 = self._build_dqn(lr,len(action_space),states_dim)

    def _build_dqn(lr,number_actions,state_dim):
        model = keras.Sequential([
            keras.layers.Dense(state_dim,activation='relu'),
            keras.layers.Dense(128,activation='tanh'),
            keras.layers.Dense(64,activation='tanh'),
            keras.layers.Dense(number_actions,activation=None)
            ])
        model.compile(optimizer=Adam(learning_rate=lr),loss='mean_squared_error')
        #model.summary()
        return model

    def save_transition(self,state,action,reward,state_next,is_terminal):
      self.memory.save_transition(state,action,reward,state_next,is_terminal)

    def calculate_epsilon(self):
      discount_eps=1
      if self.epsilon <4*self.epsilon_end:
        discount_eps=0.25
      if self.epsilon <2*self.epsilon_end:
        discount_eps=0.025
      self.epsilon = max(self.epsilon - discount_eps*self.epsilon_dec,self.epsilon_end)
    
    def calculate_lr(self):
      print("lr to 0.95*lr")
      self.lr=0.95*self.lr
      K.set_value(self.q_eval1.optimizer.learning_rate,self.lr )
      K.set_value(self.q_eval2.optimizer.learning_rate,self.lr )


    def save_weights(self):
      self.q_eval1.save_weights('ddqn1_weights.h5f', overwrite=True)
      self.q_eval2.save_weights('ddqn2_weights.h5f', overwrite=True)


    def choose_action(self,observation):
    #chose random with epsilon prob 
      if np.random.random() < self.epsilon:
        return np.random.choice(self.action_space)

      state = np.array([observation])
      choose_nn1=False

      if np.random.random()<0.5:
        choose_nn1=True     
      if choose_nn1:
          return np.argmax(self.q_eval1.predict(state))
      return np.argmax(self.q_eval2.predict(state))     

    def train2nn(self,nn1,nn2,states, actions, rewards, states_next,non_terminal):
      q_eval = nn1.predict(states)
      q_next = nn2.predict(states_next)
      q_target = np.copy(q_eval)
      batch_idx = np.arange(self.batch_size, dtype=np.int32)
      q_target[batch_idx,actions] = rewards+self.gemma*np.max(q_next,axis=1)*(non_terminal)
      nn1.train_on_batch(states,q_target)

    def learn(self):
      if self.memory.memory_counter<self.batch_size:
        print("Return::batch size bigger than memory counter :batch size={0} memory_counter={1}".format(self.batch_size,self.memory.memory_counter))
        return
      
      states, actions, rewards, states_next,is_terminal = self.memory.sample_batch(self.batch_size)
      non_terminal = np.where(is_terminal==1,0,1)
 
      choose_nn1=False
      if np.random.random()<0.5:
        choose_nn1=True     
      if choose_nn1:
          self.train2nn(self.q_eval1,self.q_eval2,states, actions, rewards, states_next,non_terminal)
          return                   
      self.train2nn(self.q_eval2,self.q_eval1,states, actions, rewards, states_next,non_terminal)

print("start")
#ag = dummy_agent()

mem = ReplayMemory(MEMORY_SIZE, actions_dim, states_dim)

# ag_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,,epsilon=1,epsilon_dec=0.005,epsilon_end=0.04,alpha=lr,
#                beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=256,layer2_size=256)
# ag_no_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,action_space=action_space_vectors,epsilon=0,epsilon_dec=0,epsilon_end=0,alpha=lr,
#                beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=256,layer2_size=256)
# ag_half_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,action_space=action_space_vectors,epsilon=0.51,epsilon_dec=0.0025,epsilon_end=0.04,alpha=lr,
#                beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=4096,layer2_size=256)

#train_loop(ag, 2000, env)
#last one was with eps=0

#logging.info("start ag_no_eps train")
#train_loop(ag_no_eps, 2000, env)
#logging.info("done with ag_no_eps train")

# logging.info("start ag_eps train")
# train_loop(ag_eps, 2000, env)
# logging.info("done with ag_eps train")
#
# logging.info("start ag_half_eps train")
# train_loop(ag_half_eps, 2000, env)
# logging.info("done with ag_half_eps train")
# action_discretizationer= Discretizationer(NUM_ACTION_BINS, -1, 1)

ac_agent = ActorCritic(memory=mem,batch_size=BATCH_SIZE,input_len=states_dim, actions_dim=actions_dim, index_to_action=env.index_to_action,
                        epsilon=EPSILON,epsilon_dec=EPSILON_DEC_RATE,epsilon_end=EPSILON_MIN,alpha=lr_low1,
                        beta=lr_low2,gamma=GAMMA,clip_value=CLIP_VALUE,layer1_size=LAYER1_SIZE,layer2_size=LAYER2_SIZE)

logging.info("start ag_half_eps lr low train")
train_loop(ac_agent, NUM_EPOCHS, env)
logging.info("done with ag_half_eps lr low train")

