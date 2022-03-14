import numpy as np
import torch
import random
from collections import deque

class ReplayMemory:
    def __init__(self, max_size, actions_dim, state_dim):
        self.memory_size = max_size
        self.memory_counter = 0

        # self.actions_dim = np.prod(NUM_ACTION_BINS)
        
        self.states = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        self.states_next = np.zeros((self.memory_size, state_dim), dtype=np.float32)
        # action space discrete
        # self.actions_idx = np.zeros((self.memory_size, self.actions_dim),dtype=int)
        self.actions_idx = np.zeros((self.memory_size),dtype=int)
        self.actions = np.zeros((self.memory_size, actions_dim),dtype=float)
        # rewards are floatType
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        # boolean but will be represeted as int
        self.is_terminal = np.zeros(self.memory_size, dtype=np.int32)

    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        idx = self.memory_counter % self.memory_size
        self.states[idx] = state
        self.states_next[idx] = state_next
        # self.actions_idx[idx, action_idx] = 1.0
        self.actions_idx[idx] = action_idx
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
    
    def __len__(self):
        return self.memory_counter

class TensorReplayMemory:
    def __init__(self, max_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen=max_size)

    def save_transition(self, state, action_idx, action, reward, state_next, is_terminal):
        # add samples to replay memory
        self.buffer.append((state, action_idx, action, reward, state_next, is_terminal))

    def sample_batch(self, batch_size):
        # random samples from replay memory
        mini_batch = random.sample(self.buffer, batch_size)
        states = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
        actions_idx = torch.from_numpy(np.vstack([i[1] for i in mini_batch]).astype(np.uint8)).float().to(self.device)
        actions = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
        states_next = torch.from_numpy(np.vstack([i[4] for i in mini_batch])).float().to(self.device)
        is_terminal = torch.from_numpy(np.vstack([i[5] for i in mini_batch]).astype(np.uint8)).float().to(self.device)
        return states, actions_idx, actions, rewards, states_next, is_terminal

    def __len__(self):
        return len(self.buffer)