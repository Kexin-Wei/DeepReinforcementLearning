# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import random

from collections import deque

from tensorflow.python.ops.gen_array_ops import dequantize
from dm_wrapper import make_env


# %%
class DDQN:
    # Reference:
    # https://github.com/EvolvedSquid/tutorials/blob/master/dqn/train_dqn.ipynb
    def __init__(self, N_ACT,LEARNING_RATE = 0.01):
        self.INPUT_SIZE = (84,84,4)
        self.N_ACT      = N_ACT

        self.LEARNING_RATE = LEARNING_RATE
        self.model = self.create_dqn()

    def create_dqn(self):
        input_layer = tf.keras.layers.Input(shape=self.INPUT_SIZE)
        cnn1 = tf.keras.layers.Conv2D(32,8,strides=4,activation='relu')(input_layer)
        cnn2 = tf.keras.layers.Conv2D(64,4,strides=2,activation='relu')(cnn1)
        cnn3 = tf.keras.layers.Conv2D(64,3,strides=1,activation='relu')(cnn2)
        flatten = tf.keras.layers.Flatten()(cnn3)
        value = tf.keras.layers.Dense(1)(flatten)
        advantage = tf.keras.layers.Dense(self.N_ACT)(flatten)

        reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True)) # custom a type of layer
        q_output = tf.keras.layers.Add()([
            value,tf.keras.layers.Subtract()([
                advantage,reduce_mean(advantage)
                ])
            ])
        model = tf.keras.Model(input_layer,q_output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss = tf.keras.losses.Huber(),
            metrics = ['accuracy']
        )
        return model

# %%
# TODO: change replay to prioried replay

class PReplay:
    pass
# %%
# TODO: Change Agent class 
class Agent(PReplay,DDQN):
    def __init__(self, N_ACT,N_OB,                 \
                    GAMMA   = 0.9,                 \
                    EPSILON = 0.3,                 \
                    EPSILON_DECAY = 0.997,         \
                    MODEL_UPDATE_STEP   = 200,    \
                    MEMORY_SAMPLE_START = 20,     \
                    LEARNING_RATE = 0.01,          \
                    MEMORY_SIZE  = 10_000,         \
                    BATCH_SIZE   = 64,             \
                    WRAPPER_SIZE = 4 ):

        Replay.__init__(self, MEMORY_SIZE = MEMORY_SIZE, \
                        BATCH_SIZE = BATCH_SIZE)
        
        DDQN.__init__(self, N_ACT, \
                     LEARNING_RATE = LEARNING_RATE)

        #self.N_ACT   = N_ACT
        self.N_OB    = N_OB
        self.GAMMA   = GAMMA
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY

        self.target_model = self.create_cnn()
        self.target_model.set_weights(self.model.get_weights())

        self.MODEL_UPDATE_STEP = MODEL_UPDATE_STEP
        self.STEP = 0
        
        self.MEMORY_SAMPLE_START = MEMORY_SAMPLE_START
        
    
    def get_q_value(self, state):
        # state is obwrapper.packup

        return self.model.predict(state.packup())
    
    
    def get_action(self,state): # get action with epsilon greedy
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.N_ACT),0
            
        q = self.get_q_value(state)
        return np.argmax(q), np.amax(q)
    
    
    def train(self): 
        #if the momery len > 0.2 memory size
        if self.memo_len() < self.MEMORY_SAMPLE_START:
            return
        batch_memo = self.sample()
        
        # model for q now
        batch_state = np.array([ a_set_memo[0][0,:,:,:] for a_set_memo in batch_memo])
        batch_q     = self.model.predict(batch_state)
        
        # target_model for max q
        batch_state_next = np.array([ a_set_memo[3][0,:,:,:] for a_set_memo in batch_memo])
        batch_q_next = self.target_model.predict(batch_state_next)
        
        batch_q_new = []
        for index,(state, action, reward, state_next, done) in enumerate(batch_memo):
            if done:
                q_new = reward
            else:
                q_new = reward + self.GAMMA * max(batch_q_next[index])
            
            q = batch_q[index]
            q[action] = q_new
            batch_q_new.append(q)
            
        self.STEP +=1
        history = self.model.fit(batch_state,np.array(batch_q_new),batch_size = self.BATCH_SIZE, verbose = 0)
        return history.history
        
    def target_model_update(self):
        if self.STEP < self.MODEL_UPDATE_STEP:
            return
        self.STEP = 0
        self.target_model.set_weights(self.model.get_weights())
# TODO: Change 

#%%

# %% Test for DDQN

input_layer = tf.keras.layers.Input(shape=(84,84,4))
cnn1 = tf.keras.layers.Conv2D(32,8,strides=4,activation='relu')(input_layer)
cnn2 = tf.keras.layers.Conv2D(64,4,strides=2,activation='relu')(cnn1)
cnn3 = tf.keras.layers.Conv2D(64,3,strides=1,activation='relu')(cnn2)
flatten = tf.keras.layers.Flatten()(cnn3)
value = tf.keras.layers.Dense(1)(flatten)
advantage = tf.keras.layers.Dense(4)(flatten)

reduce_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True)) # custom a type of layer
q_output = tf.keras.layers.Add()([value,\
                                tf.keras.layers.Subtract()([advantage,reduce_mean(advantage)])])
model = tf.keras.Model(input_layer,q_output)

model.summary() 

# %%
env = make_env('BreakoutNoFrameskip-v4')
ob = env.reset()
ob_numpy = ob.concatenate()
print(ob_numpy.shape)
print(model.predict(tf.expand_dims(ob_numpy,0)))

# %% [markdown]
### tf.expand_dims()

#%%
print(tf.expand_dims(ob_numpy,0).shape)
# %% [markdown]
### tf.split()

# %%
x=np.random.randint(5,size=[3,3,3])
print(x.shape)
y=tf.split(x,3,0)
print(len(y))
print(y[0].shape)
print(y[1].shape)
print(y[2].shape)
# %% [markdown]
### tf.math.reduce_mean()

# %%
x=np.random.randint(6,size=[5,1])
print(x)
y=tf.reduce_mean(x,keepdims= True)
print(y)
y_x= tf.keras.Sub
# %%
import random
l= 10
batch_index = random.sample(range(l),3)
print(batch_index)
# %%
from collections import deque
import numpy as np
state=deque(range(l),maxlen = 10)
print(state)
print(np.array(state)[batch_index])

# %%
