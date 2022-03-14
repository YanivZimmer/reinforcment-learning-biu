# imports
print("start imports")
from pyclbr import Function
import gym
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

os.environ['LANG'] = 'en_US'
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
from custom_types import *
from env_wrappers import *
from replay_memory import ReplayMemory

# Disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Constants
TRMINATE_REWARD = -100
UPDATE_SECOND_NET_INTERVAL = 10
# When on dry run, no output files are created
DRY_RUN = True

# Run settings
OPENAI_ENV = EnvType.LUNAR_LANDER_CONTINUOUS_V2.value
SUCCESS_REWARD = 200 if OPENAI_ENV == EnvType.LUNAR_LANDER_CONTINUOUS_V2.value else 300
MODEL = ModelType.TD3
NUM_EPOCHS = 700
MAKE_ACTION_DISCRETE = True
NUM_ACTION_BINS = [10, 10]
NUM_ACTIONS = np.prod(NUM_ACTION_BINS)
MAKE_STATE_DISCRETE = False
NUM_STATE_BINS = 5
MEMORY_SIZE = 1000000
BATCH_SIZE = 1024
LAYER1_SIZE = 32
LAYER2_SIZE = 64
LAYER1_ACTIVATION = ActivationType.RELU.value
LAYER2_ACTIVATION = ActivationType.RELU.value
LAYER1_LOSS = LossType.HUBER.value
LAYER2_LOSS = LossType.HUBER.value
EPSILON = 0.9
EPSILON_MIN = 0.05
EPSILON_DEC_RATE = (EPSILON - EPSILON_MIN) / 200
GAMMA = 0.99
LEARNING_RATE = 1e-4
LEARNING_RATE_DEC_RATE = 1
lr_low1 = LEARNING_RATE
lr_low2 = LEARNING_RATE
FIT_EPOCHS = 10
CLIP_VALUE = 1e-9
MODIFY_REWARD = False
ENV_APPLY_TIME_LIMIT = False
MODIFIED_REWARD = -10 if MODIFY_REWARD else TRMINATE_REWARD

print(f'..................{MODEL.name}..................')

# Do not explore when using ActorCritic
if MODEL == ModelType.ACTOR_CRITIC or MODEL == ModelType.TD3:
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
def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    #env = TimeLimit(env, max_episode_steps=1000)
    return env

def PlotModel(episode, scores, averages):
    if DRY_RUN:
        return

    episodes = [i for i in range(0, episode + 1)]

    if str(episode)[-1:] == "0":# much faster than episode % 100
        pylab.plot(episodes, scores, 'b')
        pylab.plot(episodes, averages, 'r')
        pylab.title(f'{OPENAI_ENV} PPO training cycle {MODEL.name}\n\
                        lr = {LEARNING_RATE}, lr1 = {lr_low1}, lr2 = {lr_low2}, epsilon = ({EPSILON},{EPSILON_DEC_RATE},{EPSILON_MIN})\n\
                        layer1 = {LAYER1_SIZE}, layer2 = {LAYER2_SIZE}, action bins = {NUM_ACTION_BINS}, batch = {BATCH_SIZE}, reward = {MODIFIED_REWARD}, fit epochs = {FIT_EPOCHS}\n\
                        memory = {MEMORY_SIZE}, layer1 loss = {LAYER1_LOSS}, layer2 loss = {LAYER2_LOSS}', fontsize=8)
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

def change_reward(reward):
    if reward == TRMINATE_REWARD:
        logging.info('Failed!')
        if MODIFY_REWARD:
            logging.info(f'Changed reward to {MODIFIED_REWARD}')
            return MODIFIED_REWARD
    return reward
    # if reward>0:
    #     reward=1.5*reward
    # return reward

class RLModel:
    def __init__(self, memory, batch_size, input_len, actions_dim, index_to_action: Function,
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

    def end_epoch(self, epoch):
        self.calculate_epsilon()
        if (epoch % 100 == 0 and epoch > 400) or epoch == 20:
            self.calculate_lr()

    def save(self):
        pass

    def load(self):
        pass

class ActorCritic(RLModel):
    def __init__(self, memory, batch_size, input_len, actions_dim, index_to_action: Function, epsilon, epsilon_dec, epsilon_end,
                    alpha, beta, gamma, clip_value, layer1_size, layer2_size):
        super().__init__(memory, batch_size, input_len, actions_dim, index_to_action,
                            epsilon, epsilon_dec, epsilon_end, gamma)
        self.clip_value = clip_value
        self.beta=beta
        self.alpha=alpha
        self.actor, self.critic, self.policy = self.create_network(layer1_size = layer1_size, layer2_size = layer2_size,
                                                                    num_actions = self.num_actions, alpha = alpha, beta = beta)
    def calculate_lr(self):
        print(f'lr to {LEARNING_RATE_DEC_RATE} * lr')
        self.alpha = LEARNING_RATE_DEC_RATE * self.alpha
        self.beta = LEARNING_RATE_DEC_RATE * self.beta
        K.set_value(self.actor.optimizer.learning_rate, self.alpha)
        K.set_value(self.critic.optimizer.learning_rate, self.beta)

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
        first_layer=Dense(layer1_size, activation=LAYER1_ACTIVATION, kernel_initializer= \
            tf.random_normal_initializer(stddev=0.01))(input)

        bn1 = tf.keras.layers.BatchNormalization()(first_layer)
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

        loss = LAYER1_LOSS
        if loss == LossType.CUSTOM.value:
            loss = custom_loss
        
        actor.compile(optimizer=Adam(lr=alpha), loss=loss)
        critic = Model([input], [values])
        critic.compile(optimizer=Adam(lr=beta), loss=LAYER2_LOSS)
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

        # 1 - int(done) = do not take next state into consideration if done
        target = rewards + self.gamma * critic_next_value * non_terminal
        delta = target - critic_value
        self.critic.fit(states, target, verbose=0, epochs=FIT_EPOCHS, batch_size=self.batch_size)
        self.actor.fit([states, delta], actions_idx, epochs=FIT_EPOCHS, verbose=0, batch_size=self.batch_size)


class ActorCriticTD3(RLModel):
    def __init__(self, memory, batch_size, input_len, actions_dim, index_to_action: Function, epsilon, epsilon_dec,
                 epsilon_end,
                 alpha, beta, gamma, clip_value, layer1_size, layer2_size):
        super().__init__(memory, batch_size, input_len, actions_dim, index_to_action,
                         epsilon, epsilon_dec, epsilon_end, gamma)
        self.clip_value = clip_value
        self.beta = beta
        self.alpha = alpha
        self.actor, self.critic1, self.critic2, self.policy = self.create_network(layer1_size=layer1_size, layer2_size=layer2_size,
                                                                   num_actions=self.num_actions, alpha=alpha, beta=beta)

    def calculate_lr(self):
        print(f'lr to {LEARNING_RATE_DEC_RATE} * lr')
        self.alpha = LEARNING_RATE_DEC_RATE * self.alpha
        self.beta = LEARNING_RATE_DEC_RATE * self.beta
        K.set_value(self.actor.optimizer.learning_rate, self.alpha)
        K.set_value(self.critic1.optimizer.learning_rate, self.beta)
        K.set_value(self.critic2.optimizer.learning_rate, self.beta)


    def create_network(self, layer1_size, layer2_size, num_actions, alpha, beta):
        input = Input(shape=(self.input_length,))
        # For loss calculation
        delta = Input(shape=[1])
        first_layer = Dense(layer1_size, activation=LAYER1_ACTIVATION, kernel_initializer= \
            tf.random_normal_initializer(stddev=0.01))(input)

        bn1 = tf.keras.layers.BatchNormalization()(first_layer)
        second_layer = Dense(layer2_size, activation=LAYER2_ACTIVATION, kernel_initializer= \
            tf.random_normal_initializer(stddev=0.01))(bn1)
        bn2 = tf.keras.layers.BatchNormalization()(second_layer)
        probabilities = Dense(num_actions,
                              activation='softmax')(bn2)
        values1 = Dense(1, activation='linear')(bn2)
        values2 = Dense(1, activation='linear')(bn2)

        def custom_loss(actual, prediction):
            # We clip values so we dont get 0 or 1 values
            out = Keras.clip(prediction, 1e-4, 1 - 1e-4)
            # Calculate log-likelihood
            likelihood = actual * tf.math.log(out)
            # likelihood = tf.math.log(prediction)
            loss = tf.reduce_sum(-likelihood * delta)
            return loss

        actor = Model([input, delta], [probabilities])

        loss = LAYER1_LOSS
        if loss == LossType.CUSTOM.value:
            loss = custom_loss

        actor.compile(optimizer=Adam(lr=alpha), loss=loss)
        critic1 = Model([input], [values1])
        critic1.compile(optimizer=Adam(lr=beta), loss=LAYER2_LOSS)
        critic2 = Model([input], [values2])
        critic2.compile(optimizer=Adam(lr=beta), loss=LAYER2_LOSS)

        policy = Model([input], [probabilities])
        return actor, critic1,critic2, policy

    def get_name(self):
        return "Actor Critic TD3 agent"

    def _get_actor_weight_file_name(self):
        return f'weights/{OPENAI_ENV}-actor-td3-{date_str}'

    def _get_critic_weight_file_name(self):
        return f'weights/{OPENAI_ENV}-critic-td3-1-{date_str}'

    def calculate_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def _choose_action_index(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)

        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        return np.random.choice(range(0, self.num_actions), p=probabilities)

    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        self.memory.save_transition(state, action_idx, action, reward, state_next, is_terminal)

    def save(self):
        if DRY_RUN:
            return

        self.actor.save_weights(self._get_actor_weight_file_name(), overwrite=True)
        self.critic1.save_weights(self._get_critic_weight_file_name(), overwrite=True)

    def load(self):
        self.actor.load_weights(self._get_actor_weight_file_name())
        self.critic1.load_weights(self._get_critic_weight_file_name())

    def learn(self):
        self.learn_batch()

    def learn_batch(self):
        if self.memory.memory_counter < self.batch_size:
            print("Return::batch size bigger than memory counter :batch size={0} memory_counter={1}" \
                  .format(self.batch_size, self.memory.memory_counter))
            return

        states, actions_idx, _, rewards, next_states, is_terminal = self.memory.sample_batch(self.batch_size)
        critic1_next_value = self.critic1.predict(next_states)[:, 0]
        critic1_value = self.critic1.predict(states)[:, 0]
        critic2_next_value = self.critic1.predict(next_states)[:, 0]
        critic2_value = self.critic1.predict(states)[:, 0]
        # Get min critic result for each state
        critic_min_value=np.minimum(critic1_value,critic2_value)
        critic_min_next_value=np.minimum(critic1_next_value,critic2_next_value)
        non_terminal = np.where(is_terminal == 1, 0, 1)

        # 1 - int(done) = do not take next state into consideration if done
        target = rewards + self.gamma * critic_min_next_value * non_terminal
        delta = target - critic_min_value
        self.critic1.fit(states, target, verbose=0, epochs=FIT_EPOCHS, batch_size=self.batch_size)
        self.critic2.fit(states, target, verbose=0, epochs=FIT_EPOCHS, batch_size=self.batch_size)
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
            Activation(LAYER1_ACTIVATION),
            Dense(LAYER2_SIZE),
            Activation(LAYER2_ACTIVATION),
            Dense(LAYER2_SIZE),
            Activation(LAYER2_ACTIVATION),
            Dense(LAYER2_SIZE),
            Dense(num_actions),
            Activation("softmax")
        ])

        model.compile(optimizer=Adam(lr=lr), loss=LAYER1_LOSS)
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
            action_indices = action_idx
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
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def calculate_lr(self):
        print(f'lr to {LEARNING_RATE_DEC_RATE} * lr')
        K.set_value(self.q_eval.optimizer.learning_rate, self.lr)
    
    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)
    
    def load(self):
        pass
    
    def save(self):
        pass

class AgentDDQNT(RLModel):
    def __init__(self, lr, gamma, actions_dim, index_to_action: Function, batch_size, states_dim, memory: ReplayMemory,
                 epsilon, epsilon_dec=1e-3, epsilon_end=0.01,
                 fname='dqn_model.h5'):
        super().__init__(memory, batch_size, states_dim, actions_dim, index_to_action,
                         epsilon, epsilon_dec, epsilon_end, gamma)
        self.model_file = fname
        self.lr = lr
        self.num_actions = np.prod(NUM_ACTION_BINS)
        self.action_space = [i for i in range(self.num_actions)]
        self.q_eval = self._build_dqn(lr, self.num_actions, states_dim)
        self.q_target = self._build_dqn(lr, self.num_actions, states_dim)
        self.update_second_net_interval = UPDATE_SECOND_NET_INTERVAL

    def _build_dqn(self, lr, number_actions, state_dim):
        model = keras.Sequential([
            Dense(LAYER1_SIZE, input_shape=(state_dim,)),
            Activation('relu'),
            Dense(LAYER2_SIZE),
            Activation('relu'),
            Dense(number_actions),
            Activation("softmax")
        ])

        model.compile(optimizer=Adam(lr=lr), loss=LAYER1_LOSS)
        return model

    def get_name(self):
        return "Double Deep Q Network T"

    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        self.memory.save_transition(state, action_idx, action, reward, state_next, is_terminal)

    def calculate_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)

    def calculate_lr(self):
        self.lr = LEARNING_RATE_DEC_RATE * self.lr
        logging.info(f'lr to {LEARNING_RATE_DEC_RATE} * lr ={self.lr}')
        K.set_value(self.q_eval.optimizer.learning_rate, self.lr)

    def _get_q_eval_weight_file_name(self, layer_num):
        return f'weights/{OPENAI_ENV}-eval{layer_num}-{date_str}'

    def save(self):
        if DRY_RUN:
            return

        self.q_eval.save_weights(self._get_q_eval_weight_file_name(1), overwrite=True)

    def _choose_action_index(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)

        q = self.q_eval.predict(observation)
        argm = np.argmax(q, axis=1)
        return argm[0]

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action_idx = self._choose_action_index(state)
        action = self.index_to_action(action_idx)
        return action_idx, action

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return
        states, actions_idx, _, rewards, states_next, is_terminal = self.memory.sample_batch(self.batch_size)
        non_terminal = np.where(is_terminal == 1, 0, 1)
        action_indices = actions_idx

        q_eval = self.q_eval.predict(states)

        q_next = self.q_target.predict(states_next)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        max_val = np.max(q_next, axis=1)
        q_target[batch_index, action_indices] = rewards + \
                                                self.gamma * max_val * non_terminal

        _ = self.q_eval.fit(states, q_target, verbose=0)
    
    def end_epoch(self, epoch):
        super().end_epoch(epoch)
        if epoch % self.update_second_net_interval == 0:
            logging.info("updating second net")
            self.update_second_network()

    def update_second_network(self):
        self.q_target.set_weights(self.q_eval.get_weights())


class AgentDDQN(RLModel):
    def __init__(self, lr, gamma, actions_dim, index_to_action: Function, batch_size, states_dim, memory: ReplayMemory,
               epsilon, epsilon_dec=1e-3, epsilon_end=0.01,
                fname='dqn_model.h5'):
        super().__init__(memory, batch_size, states_dim, actions_dim, index_to_action,
                            epsilon, epsilon_dec, epsilon_end, gamma)
        self.model_file = fname
        self.lr=lr
        self.num_actions = np.prod(NUM_ACTION_BINS)
        self.q_eval1 = self._build_dqn(lr, self.num_actions, states_dim)
        self.q_eval2 = self._build_dqn(lr, self.num_actions, states_dim)

    def _build_dqn(self, lr, number_actions, state_dim):
        model = keras.Sequential([
            Dense(LAYER1_SIZE, input_shape=(state_dim,)),
            Activation(LAYER1_ACTIVATION),
            Dense(LAYER2_SIZE),
            Activation(LAYER2_ACTIVATION),
            Dense(LAYER2_SIZE),
            Activation(LAYER2_ACTIVATION),
            Dense(number_actions)
        ])
        model.compile(optimizer=Adam(lr=lr), loss=LAYER1_LOSS)
        return model
    
    def get_name(self):
        return "Double Deep Q Network"

    def save_transition(self,state,action_idx,action,reward,state_next,is_terminal):
        self.memory.save_transition(state,action_idx,action,reward,state_next,is_terminal)

    def calculate_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)
    
    def calculate_lr(self):
        print(f'lr to {LEARNING_RATE_DEC_RATE} * lr')
        self.lr = LEARNING_RATE_DEC_RATE * self.lr
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
        q_eval = np.concatenate(nn1.predict(states)).reshape(BATCH_SIZE, NUM_ACTIONS)
        q_next = np.concatenate(nn2.predict(states_next)).reshape(BATCH_SIZE, NUM_ACTIONS)

        q_target = np.copy(q_eval)
        batch_idx = np.arange(self.batch_size, dtype=np.int32)

        tempa = rewards + self.gamma * np.max(q_next, axis=1) * non_terminal

        for b_idx in batch_idx:
            q_target[b_idx][actions]=tempa[b_idx]
        nn1.fit(states, q_target, verbose=0)

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
        next_observation, reward, done, _ = envir.step(action)
        steps_counter += 1

        # Check if should limit env time
        # if not ENV_APPLY_TIME_LIMIT:
        #     done = False
        #
        # # If got reward for failure or success
        # if reward == TRMINATE_REWARD or reward == SUCCESS_REWARD:
        #     done = True

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
        agent.end_epoch(episode_idx)
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

print("start")

mem = ReplayMemory(MEMORY_SIZE, NUM_ACTIONS, states_dim)

if MODEL == ModelType.ACTOR_CRITIC:
    agent = ActorCritic(memory=mem,batch_size=BATCH_SIZE,input_len=states_dim, actions_dim=actions_dim, index_to_action=env.index_to_action,
                            epsilon=EPSILON,epsilon_dec=EPSILON_DEC_RATE,epsilon_end=EPSILON_MIN,alpha=lr_low1,
                            beta=lr_low2,gamma=GAMMA,clip_value=CLIP_VALUE,layer1_size=LAYER1_SIZE,layer2_size=LAYER2_SIZE)
elif MODEL == ModelType.TD3:
    agent = ActorCriticTD3(memory=mem,batch_size=BATCH_SIZE,input_len=states_dim, actions_dim=actions_dim, index_to_action=env.index_to_action,
                            epsilon=EPSILON,epsilon_dec=EPSILON_DEC_RATE,epsilon_end=EPSILON_MIN,alpha=lr_low1,
                            beta=lr_low2,gamma=GAMMA,clip_value=CLIP_VALUE,layer1_size=LAYER1_SIZE,layer2_size=LAYER2_SIZE)
elif MODEL == ModelType.DDQN:
    agent = AgentDDQN(lr=LEARNING_RATE, gamma=GAMMA, actions_dim=actions_dim, index_to_action=env.index_to_action, batch_size=BATCH_SIZE,
                        states_dim=states_dim, memory=mem, epsilon=EPSILON, epsilon_dec=EPSILON_DEC_RATE, epsilon_end=EPSILON_MIN)
elif MODEL == ModelType.DQN:
    agent = AgentDQN(lr=LEARNING_RATE, gamma=GAMMA, actions_dim=actions_dim, index_to_action=env.index_to_action, batch_size=BATCH_SIZE,
                        states_dim=states_dim, memory=mem, epsilon=EPSILON, epsilon_dec=EPSILON_DEC_RATE, epsilon_end=EPSILON_MIN)
elif  MODEL == ModelType.DDQNT:
    agent = AgentDDQNT(lr=LEARNING_RATE, gamma=GAMMA, actions_dim=actions_dim, index_to_action=env.index_to_action, batch_size=BATCH_SIZE,
                        states_dim=states_dim, memory=mem, epsilon=EPSILON, epsilon_dec=EPSILON_DEC_RATE, epsilon_end=EPSILON_MIN)

logging.info("start ag_half_eps lr low train")
train_loop(agent, NUM_EPOCHS, env)
logging.info("done with ag_half_eps lr low train")

