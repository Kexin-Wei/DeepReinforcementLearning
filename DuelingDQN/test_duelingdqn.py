# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
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
            loss = tf.keras.losses.Huber()
        )
        return model

# %%
# TODO: change replay to prioried replay

class Replay:
    def __init__(self, MEMORY_SIZE = 5000, BATCH_SIZE = 64):
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.memory = deque([],maxlen = MEMORY_SIZE)
        
    def memo_append(self, a_set_memory):
        # a_set_memory = sars(a) : [ob, (act), reward, ob_next, done]
        self.memory.append(a_set_memory)

    def memo_len(self):
        return len(self.memory)
        
    def sample(self):
        return random.sample(self.memory,self.BATCH_SIZE)

# %%
# TODO: Change Agent class 
# TODO: Change 
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
