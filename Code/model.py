import tensoflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as Keras

class Actor:
    def __init__(self,input_shape,action_space,lr,
    optimizer,ly1=512,ly2=256,ly3=64):
        self.action_space = action_space
        actor_input=Input(input_shape)
        A=Dense(ly1,activation='relu',kernel_initializer=\
        tf.random_normal_initializer(stddev=0.01))(actor_input)
        if ly2>0:
            A=Dense(ly2,activation='relu',kernel_initializer=\
                tf.random_normal_initializer(stddev=0.01))(A)

        if ly3>0:
            A=Dense(ly3,activation='relu',kernel_initializer=\
                tf.random_normal_initializer(stddev=0.01))(A)
        
        output=Dense(self.action_space,activation="softmax")(A)
        self.Actor = Model(input = actor_input,output=output)
        self.Actor.compile(optimizer=optimizer(lr=lr),loss=self.ppo_loss)    

    def ppo_loss(self,y_true,y_pred):
        advantages = y_true[:,:1]
        prediction_picks =  y_true[:, 1:1+self.action_space]
        actions = y_true[:, 1+self.action_space:]
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

    def predict(selt,state):
        return selt.Actor.predict(state)
