
import os
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from model import AC_FC

# %%
class Logfile:
    def __init__(self, DIR):
        self.DIR = DIR
        self.log_file = open(DIR,'a')    
        
    def open(self):
        self.log_file = open(self.DIR,'a')
        
    def write(self,string):
        print(string,end="")        
        self.log_file.write(string)
        
    def close(self):
        self.log_file.close()
# %%
def A2C(lr,BETA,
        GAMMA,
        env,N_OB,N_ACT,
        fc1,fc2,
        EPOCHS,DIR,
        TENSORBOARD_FLAG=False,RENDER_FLAG=False):
    # RENDER_FLAG = False
    # TENSORBOARD_FLAG = True
    # EPOCHS = 3000
    # # %%
    # env_name = "LunarLander-v2"
    # env = gym.make(env_name)

    # N_OB  = env.observation_space.shape[0]
    # N_ACT = env.action_space.n
    
    ACNet = AC_FC(lr,N_OB, N_ACT,
                  FC1_DIMS = fc1, FC2_DIMS = fc2)
    
    comment = f"lr_{lr}_fc1_{fc1}_fc2_{fc2}"

    if TENSORBOARD_FLAG:
        writer = SummaryWriter(os.path.join(DIR,comment))
    TEST = False
    ep=0
    reward_list = []
    # %%
    def get_action(ob):    
        v, p = ACNet.forward(ob)
        dist = torch.distributions.Categorical(p) # discrete distribution with N_ACTs probabilities
        act  = dist.sample()
        return act.item()
    
    def train(ob, act, reward, next_ob, done):
        v,p = ACNet.forward(ob)
        next_v , next_p = ACNet.forward(next_ob)

        act_tensor = torch.Tensor([act]).to(ACNet.device)
        # TD error
        td_error  = torch.Tensor([reward]).to(ACNet.device)\
                    + (1-done)*GAMMA* next_v - v
                    
        # actor loss <- negative *log(p(ob,act)*TD error
        dist = torch.distributions.Categorical(p) # discrete distribution with N_ACTs probabilities
        actor_loss   = - dist.log_prob(act_tensor) * td_error
        entropy_loss = - BETA*dist.entropy() # already has - 
        critic_loss  = td_error**2
        
        ACNet.optimizer.zero_grad()
        (actor_loss+entropy_loss+critic_loss).backward()
        ACNet.optimizer.step()
    # %%
    while(1):
        ob = env.reset()
        
        reward_sum = 0
        
        if TEST: images = []
        #===============================
        while(1):
            
            if RENDER_FLAG:
                env.render()
            if TEST:
                images.append(env.render(mode='rgb_array'))
                
            act = get_action(ob)
            
            next_ob, reward, done, info = env.step(act)
            
            train(ob, act, reward, next_ob, done)
            
            ob = next_ob
            reward_sum += reward
            
            if done:
                break
        #========================            
        if TEST:
            print(f"Final Test: reward:{reward_sum}")
            break
        
        ep += 1
        reward_list.append(reward_sum)
        
        print(f"Epoch:{ep} \t reward: {reward_sum:.2f} \t best reward:{max(reward_list):.2f}")
        
        if TENSORBOARD_FLAG:
            writer.add_scalar("Reward",reward_sum,ep)
            writer.flush()
        
        if ep>EPOCHS:
            TEST = True
        
    # %%
    if TEST:
        imageio.mimsave(f"{DIR}/{comment}_{reward_sum}.gif",images,fps=50)
        
    if TENSORBOARD_FLAG:
        writer.close()

    plt.figure()
    plt.plot(reward_list)
    plt.title("Reward")
    plt.savefig(f"{DIR}/{comment}_reward_max{max(reward_list):.2f}.png")
    env.close()

def A2C_GAE(lr,
            GAMMA,LAMBDA,
            COEF_VALUE,COEF_ENT,
            EPOCHS,DIR,
            env,N_OB, N_ACT,
            fc1,fc2,
            TENSORBOARD_FLAG=False,RENDER_FLAG=False):
    
    ACNet = AC_FC(lr, N_OB, N_ACT,
              FC1_DIMS = fc1, FC2_DIMS = fc2)
    
    comment = f"lr_{lr}_fc1_{fc1}_fc2_{fc2}"
    
    def get_action(ob):    
        v, p = ACNet.forward(ob)
        dist = torch.distributions.Categorical(p) # discrete distribution with N_ACTs probabilities
        act  = dist.sample()
        return act.item()

    def batchrize(memory):
        obs, acts, rewards, next_obs, dones = zip(*[memory[idx] for idx in range(len(memory))])
        return np.array(obs), np.array(acts).astype(int), \
            np.array(rewards), np.array(next_obs) , np.array(dones)
            

    def train(memory):
        obs, acts, rewards, next_obs, dones = batchrize(memory)
        value_s,p_s = ACNet.forward(obs)
        value_next_s, p_next_s = ACNet.forward(next_obs)
        
        
        acts    = torch.Tensor(acts).to(ACNet.device)
        rewards = torch.Tensor(rewards).to(ACNet.device)
        dones   = torch.Tensor(dones).to(ACNet.device)
        
        td_errors = rewards + (1-dones)*GAMMA*value_next_s[0]-value_s[0]
        advantage = 0
        for i in reversed(range(td_errors.shape[0])):
            advantage = LAMBDA*GAMMA*advantage+td_errors[i]
            
        dist = torch.distributions.Categorical(p_s)
        # sum or mean?
        actor_loss   = -(dist.log_prob(acts)*advantage).mean()
        entropy_loss = -COEF_ENT*dist.entropy().mean()
        critic_loss  = COEF_VALUE*(td_errors**2).mean()
        
        ACNet.optimizer.zero_grad()
        (actor_loss+entropy_loss+critic_loss).backward()
        ACNet.optimizer.step()
    
    if TENSORBOARD_FLAG:
        writer = SummaryWriter(os.path.join(DIR,comment))
        
    TEST = False
    reward_list = []
    ep=0
    memo = namedtuple('Experience', ['state','act','reward','next_state','done'])
    #%%time
    while(1):
        ob = env.reset()
        reward_sum = 0
        
        memory = []
        
        if TEST: images = []
        #=======================================================
        while(1):
            
            if RENDER_FLAG:
                env.render()
            if TEST:
                images.append(env.render(mode='rgb_array'))
                
            act = get_action(ob)
            
            next_ob, reward, done, info = env.step(act)
            
            #train(ob, act, reward, next_ob, done)
            memory.append(memo(ob,act,reward,next_ob,done))
            
            ob = next_ob
            reward_sum += reward
                
            if done:
                
                break
        #==================================================
        train(memory)
        
        if TEST:
            print(f"Final Test: reward:{reward_sum}")
            break
        
        ep += 1
        reward_list.append(reward_sum)
        
        print(f"Epoch:{ep} \t reward:{reward_sum:.2f} \t  best reward:{max(reward_list):.2f}")
        
        if TENSORBOARD_FLAG:
            writer.add_scalar("Reward",reward_sum,ep)
            writer.flush()
        
        if ep>EPOCHS:
            TEST = True
        
    # %%
    if TEST:
        imageio.mimsave(f"{DIR}/{comment}_reward_{reward_sum}.gif",images,fps=50)

    if TENSORBOARD_FLAG:
        writer.close()

    plt.figure()
    plt.plot(reward_list)
    plt.title("Reward")
    plt.savefig(f"{DIR}/{comment}_reward_max_{max(reward_list):.2f}.png")

    env.close()