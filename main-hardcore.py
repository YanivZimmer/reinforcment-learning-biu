# Based on: https://github.com/honghaow/FORK/blob/master/BipedalWalkerHardcore/TD3_FORK_BipedalWalkerHardcore_Colab.ipynb

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.optim as optim
import logging
import os
from tensorflow.keras.layers import Dense
import numpy as np
from datetime import datetime
import pylab
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyclbr import Function
from custom_types import *
from env_wrappers import *
from replay_memory import TensorReplayMemory

os.environ['LANG'] = 'en_US'

# Constants
TRMINATE_REWARD = -100
UPDATE_SECOND_NET_INTERVAL = 10
# When on dry run, no output files are created
DRY_RUN = False

# Run settings
OPENAI_ENV = EnvType.BIPEDAL_WALKER_HARDCORE.value
SUCCESS_REWARD = 200 if OPENAI_ENV == EnvType.LUNAR_LANDER_CONTINUOUS_V2.value else 300
MODEL = ModelType.TD3
NUM_EPOCHS = 100000
MAX_STEPS = 3000
MAKE_ACTION_DISCRETE = False
NUM_ACTION_BINS = [4, 4, 4, 4]
NUM_ACTIONS = np.prod(NUM_ACTION_BINS)
MAKE_STATE_DISCRETE = False
NUM_STATE_BINS = 5
MEMORY_SIZE = 1000000
BATCH_SIZE = 100
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LAYER1_ACTIVATION = ActivationType.RELU.value
LAYER2_ACTIVATION = ActivationType.RELU.value
LAYER1_LOSS = LossType.MSE.value
LAYER2_LOSS = LossType.MSE.value
EPSILON = 0.9
EPSILON_MIN = 0.05
EPSILON_DEC_RATE = (EPSILON - EPSILON_MIN) / 200
GAMMA = 0.99
TAU = 0.02
LEARNING_RATE = 3e-4
LEARNING_RATE_DEC_RATE = 1
FIT_EPOCHS = 10
CLIP_VALUE = 1e-9
MODIFY_REWARD = False
ENV_APPLY_TIME_LIMIT = False
MODIFIED_REWARD = -10 if MODIFY_REWARD else TRMINATE_REWARD
SEED = 88

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
                        lr = {LEARNING_RATE}, epsilon = ({EPSILON},{EPSILON_DEC_RATE},{EPSILON_MIN})\n\
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

# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, layer1_size=400, layer2_size=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return F.torch.tanh(self.layer3(x))


# Q1-Q2-Critic Neural Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, layer1_size=400, layer2_size=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Q1 architecture
        self.layer1 = nn.Linear(state_size + action_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, 1)

        # Q2 architecture
        self.layer4 = nn.Linear(state_size + action_size, layer1_size)
        self.layer5 = nn.Linear(layer1_size, layer2_size)
        self.layer6 = nn.Linear(layer2_size, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.layer1(xa))
        x1 = F.relu(self.layer2(x1))
        x1 = self.layer3(x1)

        x2 = F.relu(self.layer4(xa))
        x2 = F.relu(self.layer5(x2))
        x2 = self.layer6(x2)

        return x1, x2

class SysModel(nn.Module):
    def __init__(self, state_size, action_size, layer1_size=400, layer2_size=300):
        super(SysModel, self).__init__()
        self.layer1 = nn.Linear(state_size + action_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, state_size)

    def forward(self, state, action):
        """Build a system model to predict the next state at a given state."""
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.layer1(xa))
        x1 = F.relu(self.layer2(x1))
        x1 = self.layer3(x1)

        return x1

class TD3_FORK(RLModel):
    def __init__(
            self, name, env,
            memory,
            gamma=GAMMA,  # discount factor
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            tau=TAU,  # target network update factor
            policy_noise=0.2,
            std_noise=0.1,
            noise_clip=0.5,
            policy_freq=2  # target network update period
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.lr = lr
        self._create_models()
        self._define_optimizers()
        self.save()
        self.memory = memory
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.name = name
        self.upper_bound = self.env.action_space.high[0]  # action space upper bound
        self.lower_bound = self.env.action_space.low[0]  # action space lower bound
        self.obs_upper_bound = self.env.observation_space.high[0]  # state space upper bound
        self.obs_lower_bound = self.env.observation_space.low[0]  # state space lower bound
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise

    def _create_models(self):
        self.create_actor()
        self.create_critic()
        self.create_sysmodel()
    
    def _define_optimizers(self):
        self.act_opt = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.sys_opt = optim.Adam(self.sysmodel.parameters(), lr=self.lr)

    def create_actor(self):
        params = {
            'state_size': self.env.observation_space.shape[0],
            'action_size': self.env.action_space.shape[0],
            'seed': SEED
        }
        self.actor = Actor(**params, layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE).to(self.device)
        self.actor_target = Actor(**params, layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE).to(self.device)

    def create_critic(self):
        params = {
            'state_size': self.env.observation_space.shape[0],
            'action_size': self.env.action_space.shape[0],
            'seed': SEED
        }
        self.critic = Critic(**params, layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE).to(self.device)
        self.critic_target = Critic(**params, layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE).to(self.device)

    def create_sysmodel(self):
        params = {
            'state_size': self.env.observation_space.shape[0],
            'action_size': self.env.action_space.shape[0]
        }
        self.sysmodel = SysModel(**params, layer1_size=LAYER1_SIZE, layer2_size=LAYER2_SIZE).to(self.device)

    def get_name(self):
        return "Actor Critic TD3 agent"
    
    def save(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def load(self):
        self.actor.load_state_dict(
            torch.load('/content/drive/My Drive/bipedal/weights/hardcore/actor.pth', map_location=self.device))
        self.critic.load_state_dict(
            torch.load('/content/drive/My Drive/bipedal/weights/hardcore/critic.pth', map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load('/content/drive/My Drive/bipedal/weights/hardcore/actor_t.pth', map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load('/content/drive/My Drive/bipedal/weights/hardcore/critic_t.pth', map_location=self.device))
        self.sysmodel.load_state_dict(
            torch.load('/content/drive/My Drive/bipedal/weights/hardcore/sysmodel.pth', map_location=self.device))
    
    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        self.memory.save_transition(state, action_idx, action, reward, state_next, is_terminal)

    def learn_and_update_weights_by_replay(self, training_iterations, weight, totrain):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        # print(len(self.replay_memory_buffer))
        if len(self.memory) < 1e4:
            return 1
        for it in range(training_iterations):
            states, actions_idx, actions, rewards, states_next, is_terminal = self.memory.sample_batch(self.batch_size)

            # Training and updating Actor & Critic networks.

            # Train Critic
            target_actions = self.actor_target(states_next)
            offset_noises = torch.FloatTensor(actions.shape).data.normal_(0, self.policy_noise).to(self.device)

            # clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(self.lower_bound, self.upper_bound)

            # Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(states_next, target_actions)
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = rewards + self.gamma * Q_targets * (1 - is_terminal)

            # Compute current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            self.soft_update_target(self.critic, self.critic_target)

            # Train_sysmodel
            predict_next_state = self.sysmodel(states, actions) * (1 - is_terminal)
            states_next = states_next * (1 - is_terminal)
            sysmodel_loss = F.mse_loss(predict_next_state, states_next.detach())
            self.sys_opt.zero_grad()
            sysmodel_loss.backward()
            self.sys_opt.step()

            s_flag = 1 if sysmodel_loss.item() < 0.020 else 0

            # Train Actor
            # Delayed policy updates
            if it % self.policy_freq == 0 and totrain == 1:
                actions = self.actor(states)
                actor_loss1, _ = self.critic_target(states, actions)
                actor_loss1 = actor_loss1.mean()
                actor_loss = - actor_loss1

                if s_flag == 1:
                    p_actions = self.actor(states)
                    p_next_state = self.sysmodel(states, p_actions).clamp(self.obs_lower_bound,
                                                                               self.obs_upper_bound)

                    p_actions2 = self.actor(p_next_state.detach()) * self.upper_bound
                    actor_loss2, _ = self.critic_target(p_next_state.detach(), p_actions2)
                    actor_loss2 = actor_loss2.mean()

                    p_next_state2 = self.sysmodel(p_next_state.detach(), p_actions2).clamp(self.obs_lower_bound,
                                                                                           self.obs_upper_bound)
                    p_actions3 = self.actor(p_next_state2.detach()) * self.upper_bound
                    actor_loss3, _ = self.critic_target(p_next_state2.detach(), p_actions3)
                    actor_loss3 = actor_loss3.mean()

                    actor_loss_final = actor_loss - weight * (actor_loss2) - 0.5 * weight * actor_loss3
                else:
                    actor_loss_final = actor_loss

                self.act_opt.zero_grad()
                actor_loss_final.backward()
                self.act_opt.step()

                # Soft update target models

                self.soft_update_target(self.actor, self.actor_target)

        return sysmodel_loss.item()

    def soft_update_target(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def policy(self, state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        # Adding noise to action
        shift_action = np.random.normal(0, self.std_noise, size=self.env.action_space.shape[0])
        sampled_actions = (actions + shift_action)
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return np.squeeze(legal_action)

    def select_action(self, state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.actor_target(state).cpu().data.numpy()
        return np.squeeze(actions)

    def eval_policy(self, env_name, seed, eval_episodes):
        eval_env = env_name
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward


"""Training the agent"""
gym.logger.set_level(40)
def change_reward(reward, expcount, falling_down):
    if reward == TRMINATE_REWARD:
        add_reward = -1
        reward = -5
        falling_down += 1
        expcount += 1
    else:
        add_reward = 0
        reward = 5 * reward
    return reward, add_reward, expcount, falling_down

def run_episode(env, agent, max_steps, expcount, falling_down, train_flag):
    timestep = 0
    state = env.reset()
    ep_score = 0
    #WARN maybe shoulfd be for all
    replay_buffer = []
    for _ in range(max_steps):
        #WARN maybe do random?
        action=agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        ep_score += reward
        reward, add_reward, expcount, falling_down = change_reward(reward, expcount, falling_down)
        replay_buffer.append((state, 0, action, reward, add_reward, next_state, done))
        if done:
            if add_reward == -1 or ep_score < 250:
                train_flag = 1
                for state, action_idx, action, reward, add_reward, next_state, done in replay_buffer:
                    agent.save_transition(state, action_idx, action, reward, next_state, done)
            elif expcount > 0 and np.random.rand() > 0.5:
                train_flag = 1
                expcount -= 10
                for state, action_idx, action, reward, add_reward, next_state, done in replay_buffer:
                    agent.save_transition(state, action_idx, action, reward, next_state, done)
            break

        state = next_state
        timestep += 1
        # total_timesteps += 1
    return ep_score, timestep, expcount, falling_down, train_flag

# Solve values:
# gamma=0.99,  # discount factor
# lr=3e-4,
# batch_size=100,
# buffer_capacity=1000000,
# tau=0.02,  # target network update factor
# random_seed=np.random.randint(1, 10000),
# cuda=True,
# policy_noise=0.2,
# std_noise=0.1,
# noise_clip=0.5,
# policy_freq=2  # target network update period
# batch_size = 100
# max_episodes = 100000
# max_steps = 3000
# falling_down = 0
# episodes_score = []
# avrage_reward = []
# train_flag = 0
# expcount = 0
# gym.logger.set_level(40)
# max_steps = 3000
# falling_down = 0
def train_loop(agent, episodes, max_steps, env):
    falling_down = 0
    episodes_score = []
    avrage_reward = []
    train_flag = 0
    expcount = 0
    for eps_num in range(episodes):
        score, timestep, expcount, falling_down, train_flag = run_episode(env, agent, max_steps, expcount, falling_down, train_flag)
        episodes_score.append(score)
        avg_reward = np.mean(episodes_score[-100:])
        avrage_reward.append(avg_reward)
        # If the agent's reward average is above 200
        if avg_reward > 200:
            test_reward = agent.eval_policy(env, seed=SEED, eval_episodes=10)
            # If agent gets an average of above 300 on 10 episodes
            if test_reward > 300:
                final_test_reward = agent.eval_policy(env, seed=SEED, eval_episodes=100)
                print(f'final_test_reward={final_test_reward}')
                # If agent gets an average of above 300 on 100 episodes
                if final_test_reward > 300:
                    print("*********************************Solved*********************************")
                    break

        # Training agent just on new experiences added to the replay buffer
        weight = 1 - np.clip(np.mean(episodes_score[-100:]) / 300, 0, 1)
        if train_flag == 1:
            sys_loss = agent.learn_and_update_weights_by_replay(timestep, weight, train_flag)
        else:
            sys_loss = agent.learn_and_update_weights_by_replay(100, weight, train_flag)
        train_flag = 0
        logging.info(f'episode {eps_num} score {score} avg_score {avg_reward}')
        PlotModel(eps_num, episodes_score, avrage_reward)

memory = TensorReplayMemory(MEMORY_SIZE)
agent = TD3_FORK('Bipedalhardcore', env, memory, batch_size=BATCH_SIZE, gamma=GAMMA)
train_loop(agent, NUM_EPOCHS, MAX_STEPS, env)