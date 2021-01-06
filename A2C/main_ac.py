# %%

import gym
import datetime
import os
from agent import Agent

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

RENDER_FLAG = False
TENSORBOARD_FLAG = True

# %%

env_name = "LunarLander-v2"
env = gym.make(env_name)

N_OB  = env.observation_space.shape[0]
N_ACT = env.action_space.n

agent = Agent(ALPHA=1e-5, IN_DIMS=N_OB, N_ACT=N_ACT,
                GAMMA = 0.99, FC1_DIMS = 512, FC2_DIMS = 512)

FILENAME = os.path.splitext(os.path.basename(__file__))[0]
DIR = os.path.join(f"test_{FILENAME}_{env_name}",
                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    print(f"Failed to open folder {DIR}")

# %%
reward_list = []
EPOCHS = 20

if TENSORBOARD_FLAG:
    writer = SummaryWriter(DIR)
# %%
#%%time
for ep in range(EPOCHS):
    
    ob = env.reset()
    
    reward_sum = 0
    
    while(1):
        if RENDER_FLAG:
            env.render()
        act = agent.get_action(ob)
        
        next_ob, reward, done, info = env.step(act)
        
        agent.train(ob, act, reward, next_ob, done)
        
        ob = next_ob
        reward_sum += reward
        
        if done:
            break
    
    reward_list.append(reward_sum)
    print(f"Epoch:{ep}, reward:{reward_sum}")
    
    if TENSORBOARD_FLAG:
        writer.add_scalar("Reward",reward_sum,ep)
        writer.flush()
# %%
if TENSORBOARD_FLAG:
    writer.close()
else:
    plt.figure()
    plt.plot(reward_list)
    plt.title("Reward")
    plt.savefig(DIR+'/reward.png')
env.close()
# %%