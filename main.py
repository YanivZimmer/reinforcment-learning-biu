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
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from datetime import datetime
import pylab

# Disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class EnvType(str, Enum):
    LUNAR_LANDER_CONTINUOUS_V2 = 'LunarLanderContinuous-v2'
    BIPEDAL_WALKER_V3 = 'BipedalWalker-v3'

class ModelType(Enum):
    ACTOR_CRITIC = 1
    DDQN = 2
    DQN = 3

# When on dry run, no output files are created
DRY_RUN = True

# Run settings
OPENAI_ENV = EnvType.LUNAR_LANDER_CONTINUOUS_V2
MODEL = ModelType.DQN
NUM_EPOCHS = 5000
MAKE_ACTION_DISCRETE = True
NUM_ACTION_BINS = [4, 4]
MAKE_STATE_DISCRETE = False
NUM_STATE_BINS = 5
MEMORY_SIZE = 500000
BATCH_SIZE = 1024
LAYER1_SIZE = 128
LAYER2_SIZE = 64
LAYER1_ACTIVATION = 'tanh' #'relu'
LAYER2_ACTIVATION = 'tanh' #'relu'
EPSILON = 1
# prev was EPSILON_DEC_RATE = 0.99
EPSILON_DEC_RATE = 0.998
EPSILON_MIN = 0.01
GAMMA = 0.99
LEARNING_RATE = 0.00001
lr_low1 = 0.000005 #0.35 * LEARNING_RATE
lr_low2 = 0.000005 #0.25 * LEARNING_RATE
FIT_EPOCHS = 1
CLIP_VALUE = 1e-9
MODIFY_REWARD = False
MODIFIED_REWARD = -10 if MODIFY_REWARD else -100

print(f'..................{MODEL.name}..................')

# Do not explore when using ActorCritic
if MODEL == ModelType.ACTOR_CRITIC:
    EPSILON = 0
    EPSILON_DEC_RATE = 0.00
    EPSILON_MIN = 0.00

# Get start time
date_str = datetime.today().strftime('%d-%m-%Y-%H-%M-%S')

logging_handlers = [
    logging.StreamHandler()
]

# Add file handler
if not DRY_RUN:
    file_handler = logging.FileHandler(f'logs/{OPENAI_ENV}_time_{date_str}.log')
    logging_handlers.append(file_handler)

# Logging configuration
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO, handlers=logging_handlers)

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
    # if DRY_RUN:
    #     return

    episodes = [i for i in range(0, episode + 1)]

    if str(episode)[-1:] == "0":# much faster than episode % 100
        pylab.plot(episodes, scores, 'b')
        pylab.plot(episodes, averages, 'r')
        pylab.title(f'{OPENAI_ENV} PPO training cycle {MODEL.name}\n\
                        lr = {LEARNING_RATE}, lr1 = {lr_low1}, lr2 = {lr_low2}, epsilon = ({EPSILON},{EPSILON_DEC_RATE},{EPSILON_MIN})\n\
                        layer1 = {LAYER1_SIZE}, layer2 = {LAYER2_SIZE}, action bins = {NUM_ACTION_BINS}, batch = {BATCH_SIZE}\n\
                        reward = {MODIFIED_REWARD}, fit epochs = {FIT_EPOCHS}, memory = {MEMORY_SIZE}', fontsize=8)
        pylab.ylabel('Score', fontsize=10)
        pylab.xlabel('Steps', fontsize=10)
        try:
            pylab.grid(True)
            pylab.savefig(f'plots/{OPENAI_ENV}_{MODEL.name}-{date_str}.png')
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
    if reward == -100:
        logging.info('Failed!')
        if MODIFY_REWARD:
            logging.info(f'Changed reward to {MODIFIED_REWARD}')
            return MODIFIED_REWARD
    return reward
    # if reward>0:
    #     reward=1.5*reward
    # return reward

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
        if self.memory_counter > self.memory_size:
            logging.info(f"Memory reached its limit- overriding previous operations."
                         f" memory_counter={self.memory_counter} memory_size={self.memory_size}")
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

class RLModel:
    def __init__(self, memory: ReplayMemory, batch_size, input_len, actions_dim, index_to_action: Function,
                    epsilon, epsilon_dec, epsilon_end, gamma):
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
        self.episode = 0
        self.num_actions = np.prod(NUM_ACTION_BINS) #np.power(NUM_ACTION_BINS, actions_dim)

    def get_name(self):
        pass

    def _choose_action_index(self, observation):
        pass

    def choose_action(self, observation):
        action_index = self._choose_action_index(observation)
        # action = self.action_discretizationer.idx_to_val_vec(action_index, self.actions_dim)
        action = self.index_to_action(action_index)
        return action_index, action

    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        pass

    def learn(self):
        pass

    def calculate_epsilon(self):
        pass

    def calculate_lr(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

class ActorCritic(RLModel):
    def __init__(self, memory, batch_size, input_len, actions_dim, index_to_action: Function, epsilon, epsilon_dec, epsilon_end,
                    alpha, beta, gamma, clip_value, layer1_size, layer2_size):
        super().__init__(memory, batch_size, input_len, actions_dim, index_to_action,
                            epsilon, epsilon_dec, epsilon_end, gamma)
        # self.memory = memory
        # self.batch_size = batch_size
        # self.input_length=input_len
        # self.actions_dim = actions_dim
        # self.index_to_action = index_to_action
        # self.epsilon = epsilon
        # self.epsilon_dec = epsilon_dec
        # self.epsilon_end = epsilon_end
        # # Discount factor
        # self.gamma = gamma
        self.clip_value = clip_value
        # self.episode = 0
        # self.num_actions = np.prod(NUM_ACTION_BINS) #np.power(NUM_ACTION_BINS, actions_dim)
        self.beta=beta
        self.alpha=alpha
        self.actor, self.critic, self.policy = self.create_network(layer1_size = layer1_size, layer2_size = layer2_size,
                                                                    num_actions = self.num_actions, alpha = alpha, beta = beta)
    def calculate_lr(self):
        print("lr to 0.95*lr")
        self.alpha=0.95*self.alpha
        self.beta=0.95*self.beta
        K.set_value(self.actor.optimizer.learning_rate,self.alpha )
        K.set_value(self.critic.optimizer.learning_rate,self.beta )

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

    def _choose_action_index(self, observation):
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

    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)
    
    def save(self):
        if DRY_RUN:
            return
        
        self.actor.save_weights(self._get_actor_weight_file_name(), overwrite = True)
        self.critic.save_weights(self._get_critic_weight_file_name(), overwrite = True)
    
    def load(self):
        self.actor.load_weights(self._get_actor_weight_file_name())
        self.critic.load_weights(self._get_critic_weight_file_name())

    def learn(self):
        self.learn_batch()

    def learn_batch(self):
        if self.memory.memory_counter < self.batch_size:
            print("Return::batch size bigger than memory counter :batch size={0} memory_counter={1}"\
                  .format(self.batch_size, self.memory.memory_counter))
            return

        states, actions_idx, _, rewards, next_states, is_terminal = self.memory.sample_batch(self.batch_size)
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

class AgentDQN(RLModel):
    def __init__(self, lr, gamma, actions_dim, index_to_action: Function, batch_size, states_dim, memory: ReplayMemory,
               epsilon, epsilon_dec=1e-3, epsilon_end=0.01,
                fname='dqn_model.h5'):
        super().__init__(memory, batch_size, states_dim, actions_dim, index_to_action,
                            epsilon, epsilon_dec, epsilon_end, gamma)
        self.model_file = fname
        self.lr=lr
        #   self.memory = Memory
        self.num_actions = np.prod(NUM_ACTION_BINS)
        self.q_eval = self._build_dqn(lr, self.num_actions, states_dim)
        self.action_space = [i for i in range(self.num_actions)]
    def get_name(self):
        return "Vanila Deep Q Network"

    def _build_dqn(self,lr,num_actions,states_dim):
        model = keras.Sequential([
            Dense(LAYER1_SIZE, input_shape=(states_dim,)),
            Activation('relu'),
            Dense(LAYER2_SIZE),
            Activation('relu'),
            Dense(num_actions)])

        model.compile(optimizer=Adam(lr=lr), loss='mse')
        return model
    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action_idx = np.argmax(actions)
        action = self.index_to_action(action_idx)
        return action_idx, action

    def learn(self):
        if self.memory.memory_counter > self.batch_size:
            state,action_idx, _, reward, new_state, is_terminal = \
                                          self.memory.sample_batch(self.batch_size)

            #action = tf.one_hot(action_idx,self.num_actions)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action_idx, action_values)
            non_terminal = np.where(is_terminal == 1, 0, 1)

            q_eval = self.q_eval.predict(state)

            q_next = self.q_eval.predict(new_state)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                                  self.gamma*np.max(q_next, axis=1)*non_terminal

            _ = self.q_eval.fit(state, q_target, verbose=0)
            #_ = self.q_eval.fit(state, q_target,epochs=10, verbose=0)


    def calculate_epsilon(self):
        discount_eps=1
        self.epsilon = max(self.epsilon *discount_eps * self.epsilon_dec, self.epsilon_end)

    def calculate_lr(self):
        print("lr to 0.95 * lr")
        K.set_value(self.q_eval.optimizer.learning_rate, self.lr)
        # K.set_value(self.q_eval2.optimizer.learning_rate, self.lr)
    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)
    def load(self):
        pass
    def save(self):
        pass

class AgentDDQN(RLModel):
    def __init__(self, lr, gamma, actions_dim, index_to_action: Function, batch_size, states_dim, memory: ReplayMemory,
               epsilon, epsilon_dec=1e-3, epsilon_end=0.01,
                fname='dqn_model.h5'):
        super().__init__(memory, batch_size, states_dim, actions_dim, index_to_action,
                            epsilon, epsilon_dec, epsilon_end, gamma)
        self.model_file = fname
        #   self.gamma = gamma
        #   self.actions_dim = actions_dim
        #   self.epsilon = epsilon
        #   self.epsilon_dec = epsilon_dec
        #   self.epsilon_end = epsilon_end
        #   self.batch_size = batch_size
        self.lr=lr
        #   self.memory = Memory
        self.num_actions = np.prod(NUM_ACTION_BINS)
        self.q_eval1 = self._build_dqn(lr, self.num_actions, states_dim)
        self.q_eval2 = self._build_dqn(lr, self.num_actions, states_dim)

    def _build_dqn(self, lr, number_actions, state_dim):
        model = keras.Sequential([
            keras.layers.Dense(state_dim, activation='relu'),
            keras.layers.Dense(LAYER1_SIZE, activation='tanh'),
            keras.layers.Dense(LAYER2_SIZE, activation='tanh'),
            keras.layers.Dense(number_actions, activation=None)
        ])
        model.compile(optimizer=Adam(learning_rate=lr),loss='mean_squared_error')
        #model.build((state_dim,1))
        #print(model.summary())
        return model
    
    def get_name(self):
        return "Double Deep Q Network"

    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)

    def calculate_epsilon(self):
        discount_eps=1
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_end)
    
    def calculate_lr(self):
        print("lr to 0.95 * lr")
        self.lr = 0.95 * self.lr
        K.set_value(self.q_eval1.optimizer.learning_rate, self.lr)
        K.set_value(self.q_eval2.optimizer.learning_rate, self.lr)

    def _get_q_eval_weight_file_name(self, layer_num):
        return f'weights/{OPENAI_ENV}-eval{layer_num}-{date_str}'

    def save(self):
        if DRY_RUN:
            return
        
        self.q_eval1.save_weights(self._get_q_eval_weight_file_name(1), overwrite = True)
        self.q_eval2.save_weights(self._get_q_eval_weight_file_name(2), overwrite = True)

    def load(self):
        self.q_eval1.load_weights(self._get_q_eval_weight_file_name(1))
        self.q_eval2.load_weights(self._get_q_eval_weight_file_name(2))

    def _choose_action_index(self, observation):
        #chose random with epsilon prob 
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions_dim)

        state = np.array([observation])
        choose_nn1=False

        if np.random.random() < 0.5:
            choose_nn1=True     
        if choose_nn1:
            return np.argmax(self.q_eval1.predict(state))
        return np.argmax(self.q_eval2.predict(state))

    def train2nn(self, nn1, nn2, states, actions, rewards, states_next, non_terminal):
        # Flatten 1 hot vector
        # actions = np.argmax(actions, axis=1)
        # print(f'actions: {actions}')
        # action_values = np.array(self.num_actions, dtype=np.int32)
        # action_indices = np.dot(actions, action_values)
        # print(f'action_indices: {action_indices}')
        #TODO -dont use hard coded 16!
        q_eval = np.concatenate(nn1.predict(states)).reshape(BATCH_SIZE,16)
        q_next = np.concatenate(nn2.predict(states_next)).reshape(BATCH_SIZE,16)
        #q_eval=np.stack(q_eval)
        #q_next=np.stack(q_next)
        q_target = np.copy(q_eval)
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        # print(f'rewards: {rewards}')
        # print(f'gamma: {self.gamma}')
        # print(f'q_next: {q_next} ,shape:{q_next.shape}')
        # temp1=np.max(q_next, axis=1)
        # print(f'np.max(q_next, axis=1): {temp1} ,shape:{temp1.shape}')
        # temp0=np.max(q_next, axis=0)
        # print(f'np.max(q_next, axis=0): {temp0} ,shape:{temp0.shape}')
        # print(f'non_terminal: {non_terminal}')
        # print(f'actions: {actions}')
        # print(rewards.shape)
        # print((np.max(q_next, axis=1) * non_terminal).shape)
        # print((self.gamma * np.max(q_next, axis=1)).shape)
        # print((self.gamma * np.max(q_next, axis=1) * non_terminal).shape)
        # print((rewards + self.gamma * np.max(q_next, axis=1) * non_terminal).shape)
        var=np.max(q_next, axis=1) 
        tempa = rewards + self.gamma * np.max(q_next, axis=1) * non_terminal
        # print(f'tempa: {tempa}')
        # print(f'q_target:{q_target}')
        for b_idx in batch_idx:
            # print(f'b_idx: {b_idx}')
            q_target[b_idx][actions]=tempa[b_idx]
        nn1.fit(states, q_target)

    def learn(self):
        if self.memory.memory_counter<self.batch_size:
            print("Return::batch size bigger than memory counter :batch size={0} memory_counter={1}".format(self.batch_size,self.memory.memory_counter))
            return
      
        states, actions_idx, _, rewards, states_next, is_terminal = self.memory.sample_batch(self.batch_size)
        non_terminal = np.where(is_terminal==1,0,1)

        nn1 = self.q_eval2
        nn2 = self.q_eval1

        if np.random.random() < 0.5:
            nn1 = self.q_eval1
            nn2 = self.q_eval2
                         
        self.train2nn(nn1, nn2, states, actions_idx, rewards, states_next, non_terminal)

def train_step(agent: RLModel, envir):
    observation = envir.reset()
    done = False
    score = 0
    steps_counter = 0
    while not done:
        action_idx, action = agent.choose_action(observation)
        next_observation, reward, done, info = envir.step(action)
        steps_counter += 1
        reward = change_reward(reward)
        score += reward
        # memory
        agent.save_transition(observation, action_idx, action, reward,
                              next_observation, done)
        agent.learn()
        observation = next_observation
    return score
    # ...


def train_loop(agent: RLModel, episodes: int, envir) -> None:
    score_history = []
    average_history = []
    max_score = float('-inf')
    max_average = float('-inf')
    logging.info(f'start train loop for agent {agent.get_name()}')

    for episode_idx in range(episodes):
        score = train_step(agent, envir)
        agent.calculate_epsilon()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        average_history.append(avg_score)

        if average_history[-1] > max_average:
            logging.info(f'Saving weights')
            agent.save()
        #increase batch size when there are many more actions in the memory
        if episode_idx % 250 == 0 and episode_idx > 400:
            agent.batch_size = 2 * agent.batch_size
            logging.info("Increase batch size by factor of 2")
        if (episode_idx%100 == 0 and episode_idx > 400) or (score > 100 and score > max_score) or episode_idx==20:
            agent.calculate_lr()

        max_score = max(max_score, score)
        max_average = max(max_average, avg_score)


        logging.info(
            'episode {0} score {1} avg_score {2} max_score {3} epsilon {4}'\
                .format(episode_idx, score, avg_score, max_score, agent.epsilon))
        PlotModel(episode_idx, score_history, average_history)
    
    logging.info("Training is complete")

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

if MODEL == ModelType.ACTOR_CRITIC:
    agent = ActorCritic(memory=mem,batch_size=BATCH_SIZE,input_len=states_dim, actions_dim=actions_dim, index_to_action=env.index_to_action,
                            epsilon=EPSILON,epsilon_dec=EPSILON_DEC_RATE,epsilon_end=EPSILON_MIN,alpha=lr_low1,
                            beta=lr_low2,gamma=GAMMA,clip_value=CLIP_VALUE,layer1_size=LAYER1_SIZE,layer2_size=LAYER2_SIZE)
elif MODEL == ModelType.DDQN:
    agent = AgentDDQN(lr=LEARNING_RATE, gamma=GAMMA, actions_dim=actions_dim, index_to_action=env.index_to_action, batch_size=BATCH_SIZE,
                        states_dim=states_dim, memory=mem, epsilon=EPSILON, epsilon_dec=EPSILON_DEC_RATE, epsilon_end=EPSILON_MIN)
elif MODEL == ModelType.DQN:
    agent = AgentDQN(lr=LEARNING_RATE, gamma=GAMMA, actions_dim=actions_dim, index_to_action=env.index_to_action, batch_size=BATCH_SIZE,
                        states_dim=states_dim, memory=mem, epsilon=EPSILON, epsilon_dec=EPSILON_DEC_RATE, epsilon_end=EPSILON_MIN)

logging.info("start ag_half_eps lr low train")
train_loop(agent, NUM_EPOCHS, env)
logging.info("done with ag_half_eps lr low train")

