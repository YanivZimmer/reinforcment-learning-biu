from enum import Enum

class EnvType(str, Enum):
    LUNAR_LANDER_CONTINUOUS_V2 = 'LunarLanderContinuous-v2'
    BIPEDAL_WALKER_V3 = 'BipedalWalker-v3'
    BIPEDAL_WALKER_HARDCORE = 'BipedalWalkerHardcore-v3'

class ModelType(Enum):
    ACTOR_CRITIC = 1
    DDQN = 2
    DQN = 3
    DDQNT = 4
    TD3 = 5

class LossType(Enum):
    MSE = 'mean_squared_error'
    CUSTOM = 'custom'
    HUBER = 'huber'

class ActivationType(Enum):
    RELU = 'relu'
    TANH = 'tanh'
