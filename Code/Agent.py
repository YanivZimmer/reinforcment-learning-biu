from tensorflow.keras.optimizers import Adam
from critic import Critic
from model import Actor
class PPOAgent:
    def __init__(self,env_name,lr):
        self.lr=lr
        self.optimizer = Adam
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

    def act(self,state):
        pred = self.Actor.predict(state)[0]
        actor = np.random.choice(self.action_size,p=pred) 
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, pred

    def train(self):
        state = self.env.reset()
        state = np.reshape(state,[1,self.state_size[0]])
        score = 0
        done = False

        