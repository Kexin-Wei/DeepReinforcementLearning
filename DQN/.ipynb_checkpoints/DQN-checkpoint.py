#!/usr/bin/env python
# coding: utf-8

# # Atari Breakout DQN

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import os
import datetime
import random
import numpy as np
from collections import deque

#from IPython.core.debugger import set_trace
tf.debugging.set_log_device_placement(True)

# In[2]:


env_name = 'Breakout-v0'
env = gym.make(env_name)


# In[3]:


N_ACT = env.action_space.n
N_OB  = env.observation_space.shape
print("Action space: ",N_ACT)
print("Observation space: ", N_OB)


# ## Build Model

# In[8]:


class DQN_Agent:
    def __init__(self, N_ACT, N_OB, MEMORY_SIZE = 2000,BATCH_SIZE = 32, EPSILON = 0.1, GAMMA=0.9):
        self.N_ACT   = N_ACT
        self.N_OB    = N_OB
        
        self.EPSILON = EPSILON
        self.GAMMA   = GAMMA
        
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        
        self.model  = self.create_cnn()
        
        self.target_model = self.create_cnn()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = MEMORY_SIZE)
        
    def create_cnn(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8,3, padding = 'same', activation = 'relu', input_shape = self.N_OB),
            tf.keras.layers.MaxPool2D(2, strides = 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(self.N_ACT, activation = 'linear')
        ])
        
        model.compile(
            loss = 'huber_loss',
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
            metrics   = ['accuracy']
        )
        return model
    
    #==== get q-value
    def get_q(self, ob):
        return self.model.predict(ob.reshape(-1,self.N_OB[0],self.N_OB[1],self.N_OB[2])/255.0)
    
    #==== act = take_action()
    def take_action(self,ob): 
        if np.random.rand() < self.EPSILON:
            return np.random.randint(self.N_ACT)
        q_value = self.get_q(ob)
        return np.argmax(q_value)
    
    
    #===== self.replay_memory <- add(ob,act,reward,ob_next)
    def memorize(self,a_set_memory): 
        # a_set_memory = sars(a) : [ob, act, reward, ob_next, done]
        self.replay_memory.append(a_set_memory)
    
    #==== batch train 
    def train(self):       
        
        batch_memory = random.sample(self.replay_memory,self.BATCH_SIZE)
        
        batch_ob  = np.array([ a_set_memory[0] for a_set_memory in batch_memory])/255
        
        batch_ob_next   = np.array([ a_set_memory[3] for a_set_memory in batch_memory])/255        
        batch_q_next  = self.target_model.predict(batch_ob_next)
        #set_trace()
        batch_q_new = []
        # loss = (reward+ q'-q)^2/batch_size
        for index,(ob, act, reward, ob_next, done) in enumerate(batch_memory):
            if not done:
                q_next_max = np.max(batch_q_next[index])
                q_new    = reward + self.GAMMA * q_next_max
            else:
                q_new    = reward 
            batch_q_new.append(q_new)
             
        self.model.fit(batch_ob,np.array(batch_q_new),batch_size = self.BATCH_SIZE, verbose = 0)
        
    
    #==== target_model <- model
    def target_model_update(self):
        self.target_model.set_weights(self.model.get_weights())
        


# make dir path for log and figure

# In[5]:


ROOT_DIR = "../../gym_graph"
DIR = os.path.join(ROOT_DIR,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(DIR)

try:
    os.makedirs('DQN_log')
except:
    pass
log_file = open('DQN_log/log_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),'w')


# In[ ]:


EPOCHS = 20
agent = DQN_Agent(N_ACT,N_OB)

reward_summary = {
    'max':[],
    'min':[],
    'ave':[]
}

for ep in range(EPOCHS):
    ob = env.reset()
    
    all_reward = []
    step = 0
    os.makedirs(os.path.join(DIR,str(ep)))
    while(1):
        # render monitoring
        if ep % int(EPOCHS/10) == 0: #save 10 epoch move           
            plt.imsave(os.path.join(DIR,str(ep),str(step)+'.png'),env.render(mode='rgb_array'))
        # take action
        act = agent.take_action(ob)
        
        # env step
        ob_next, reward, done, info = env.step(act)
            # reward modified
            # reward = reward if done else -1
        
        # memorize: sars(a) : [ob, act, reward, ob_next, done]
        agent.memorize([ob, act, reward, ob_next, done])
        
        # q-value update
        if len(agent.replay_memory) > (agent.MEMORY_SIZE/10):
            agent.train()            
            if step % 5 == 0:
                #set_trace()
                agent.target_model_update()
            
        ob = ob_next
        all_reward.append(reward)
        step += 1
        
        if done:
            #set_trace()
            log_file.write("Epoch {} - average rewards {} with step {}\n".format(ep,sum(all_reward)/len(all_reward),step))
            print("Epoch {} - average rewards {} with step {}\n".format(ep,sum(all_reward)/len(all_reward),step))
            reward_summary['max'].append(max(all_reward))
            reward_summary['min'].append(min(all_reward))
            reward_summary['ave'].append(sum(all_reward)/len(all_reward))
            break
            
log_file.close()


# In[ ]:


# observe the final run
ob = env.reset()
all_reward = 0
step = 0
while(1):
    os.makedirs(os.path.join(DIR,'final'))
    plt.imsave(os.path.join(DIR,'final',str(step)+'.png'),env.render(mode='rgb_array'))
    act = np.argmax(agent.model.predict(ob))
    
    ob,reward,done,infor = env.step(act)
    
    all_reward +=reward
    step +=1
    if done:
        print('Final: rewards - {}, step - {}'.format(all_reward,step))
        break
        

env.close()
print("Done")

