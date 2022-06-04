import gym
from gym import spaces
from gym.spaces.discrete import Discrete
import numpy as np

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
        assert isinstance(self.action_space, spaces.Box)

        self.low = self.action_space.low if low is None else low
        self.high = self.action_space.high if high is None else high
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
        return self.all_perms[idx]

    def action(self, action):
        binned_action = []
        for x, bins in zip(action, self.action_matrix):
            mapped_idx = np.digitize([x], bins)[0]
            if mapped_idx == len(bins):
                mapped_idx -= 1
            binned_action.append(bins[mapped_idx])
        return binned_action
    
    def reverse_action(self, action):
        return super().reverse_action(action)