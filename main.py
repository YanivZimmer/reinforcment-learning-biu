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
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,handlers=[
        logging.FileHandler('bipedal_walker_time_{0}.log'.format(ticks)),
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


# make env
env = gym.make('BipedalWalker-v3').env
#env = TimeLimit(env, max_episode_steps=1000)
env = wrap_env(env)
print(env.spec)
state_size = env.observation_space

#one hot
def to_one_hot(indeces, depth):
    return tf.one_hot(indeces, depth)

class Discretizationer:
    def __init__(self,bins_num,min,max):
        self.bins_num = bins_num
        self.min=min
        self.max=max
        self.interval_size = (max - min) / bins_num
    def val_to_bin(self,val):
        bin = np.floor((val-self.min)/self.interval_size)
        return bin
    def vec_val_to_bin(self,vec):
         bin_vec = np.zeros(len(vec))
         for i in range(len(vec)):
             bin_vec[i] = self.val_to_bin(vec[i])
         return bin_vec
    def bin_to_val(self,bin):
        #get midian value of bin
        val=(bin+0.5)*self.interval_size
        return val
    def vec_bin_to_vals(self,bin_vec):
        val_vec = np.zeros(len(bin_vec))
        for i in range(len(bin_vec)):
            val_vec[i] = self.bin_to_val(bin_vec[i])
        return val_vec
    def vec_val_to_idx(self,vec):
        sum=0
        bin_vec=self.vec_val_to_bin(vec)
        length=len(bin_vec)
        for i in range(length):
            j=np.power(self.bins_num,i)
            sum += (bin_vec[i] * j)
        return sum

def change_reward(reward):
    if reward == -100:
         return -10
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
    max_score = float('-inf')
    logging.info("start train loop for agent {0}".format(agent.get_name()))

    for episode_idx in range(episodes):
        score = train_step(agent, envir)
        agent.calculate_epsilon()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        max_score = max(max_score, score)
        logging.info(
            'episode {0} score {1} avg_score {2} max_score {3} epsilon {4}'\
                .format(episode_idx, score, avg_score, max_score, agent.epsilon))
    logging.info("Training is complete")




"""### Action space"""
# Get action min and max values
action_space_raw = env.action_space
ACTION_MIN_VAL = action_space_raw.low
ACTION_MIN_MAX = action_space_raw.high



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
    def __init__(self, memory, batch_size,input_len,action_discretizationer, epsilon,epsilon_dec, epsilon_end
                 ,alpha,beta,gamma,clip_value,layer1_size,layer2_size):
        self.memory = memory
        self.batch_size = batch_size
        self.input_length=input_len
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        # Discount factor
        self.gamma = gamma
        self.clip_value = clip_value
        self.episode = 0
        self.action_discretizationer=action_discretizationer
        self.actor, self.critic, self.policy = self.create_network(layer1_size=layer1_size,layer2_size=layer2_size,num_actions=self.num_actions,alpha=alpha,beta=beta)

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
        first_layer=Dense(layer1_size, activation='relu', kernel_initializer= \
            tf.random_normal_initializer(stddev=0.01))(input)

        bn1 = tf.keras.layers.BatchNormalization()(first_layer)
        #second_layer = Dense(layer2_size, activation='relu')(bn1)
        # second_layer=Dense(layer2_size, activation='relu', kernel_initializer= \
        #     tf.random_normal_initializer(stddev=0.01))(bn1)
        # bn2 = tf.keras.layers.BatchNormalization()(second_layer)
        probabilities = Dense(num_actions,
                              activation='softmax')(bn1)
        values = Dense(1, activation='linear')(bn1)

        def custom_loss(actual, prediction):
            # We clip values so we dont get 0 or 1 values
            #out = Keras.clip(prediction,1e-4, 1 - 1e-4)
            # Calculate log-likelihood
            #likelihood = actual * tf.math.log(out)
            likelihood = tf.math.log(prediction)
            loss = tf.reduce_sum(-likelihood * delta)
            return loss

        actor = Model([input, delta], [probabilities])
        actor.compile(optimizer=Adam(lr=alpha), loss=custom_loss )
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
            return np.random.choice(self.action_discretizationer.bin_num)

        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        #probabilities = Keras.clip(probabilities, self.clip_value, 1 - self.clip_value)
        #probabilities = np.where(probabilities > (1 - self.clip_value), (1 - self.clip_value), probabilities)
        #probabilities = np.where(probabilities < self.clip_value, self.clip_value, probabilities)
        #probabilities = np.where(np.isnan(probabilities), self.clip_value, probabilities)

        # Normalize probabilities
        probabilities = probabilities / Keras.sum(probabilities)
        #tf.random.categorical(probabilities,self.action_space_indices)
        #the_np = probabilities.eval()
        #tf.make_tensor_proto(probabilities)
        return np.random.choice(self.action_discretizationer.bin_num, p=probabilities)

    def choose_action(self, observation):
        action_index = self.choose_action_index(observation)
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
states_dim = 24
mem = ReplayMemory(5000, states_dim, True, True)
lr=0.00025

ag_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,,epsilon=1,epsilon_dec=0.005,epsilon_end=0.04,alpha=lr,
               beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=256,layer2_size=256)
ag_no_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,action_space=action_space_vectors,epsilon=0,epsilon_dec=0,epsilon_end=0,alpha=lr,
               beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=256,layer2_size=256)
ag_half_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,action_space=action_space_vectors,epsilon=0.51,epsilon_dec=0.0025,epsilon_end=0.04,alpha=lr,
               beta=lr,gamma=0.99,clip_value=1e-9,layer1_size=4096,layer2_size=256)

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

lr_low1=0.035*lr
lr_low2=0.025*lr
ag_half_eps=ActorCritic(memory=mem,batch_size=64,input_len=states_dim,action_space=action_space_vectors,epsilon=0.4,epsilon_dec=0.0005,epsilon_end=0.05,alpha=lr_low1,
               beta=lr_low2,gamma=0.99,clip_value=1e-9,layer1_size=2400,layer2_size=256)

logging.info("start ag_half_eps lr low train")
train_loop(ag_half_eps, 3000, env)
logging.info("done with ag_half_eps lr low train")

