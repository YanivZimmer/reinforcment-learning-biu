from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as Keras
import numpy as np

class Critic:
    def __init__(self, input_shape, action_space, lr,
     optimizer, ly1=512, ly2=256, ly3=64):
        critic_input = Input(input_shape)
        old_val = Input(shape=(1,))

        C = Dense(ly1,activation='relu',kernel_initializer='he_uniform')(critic_input)
        if ly2>0:
            C = Dense(ly2,activation='relu',
            kernel_initializer='he_uniform')(C)
        if ly3>0:
            C = Dense(ly3,activation='relu',
            kernel_initializer='he_uniform')(C)
        value = Dense(1,activation=None)(C)

        self.Critic = Model(input=[critic_input,old_val],outputs=value)
        self.Critic.compile(loss=[self.critic_ppo_loss(old_val)],optimizer(lr=lr))

    def critic_ppo_loss(self,values):
        def loss(y_true,y_pred):
            loss_clip = 0.2
            clip_val_loss = values + Keras.clip(y_pred-values,-loss_clip,loss_clip)
            v_loss1 = (y_true - clip_val_loss) ** 2 
            v_loss2 = (y_true-y_pred) ** 2
            val_loss = 0.5 *Keras.mean(Keras.maximum(v_loss1,v_loss2))
            return val_loss
        return loss

    def predict(self,state):
        return self.Critic.predict([state,np.zeros((state.shape[0],1))])
                   