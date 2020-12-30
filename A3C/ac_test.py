# %%
from typing import Sequence
import torch
import numpy as np
from model import ActorCritic
from dm_wrapper import make_env
# %%
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Running on GPU")
else:
    DEVICE = torch.device('cpu')
    print("Running on CPU")
    
# %%
env = make_env("PongNoFrameskip-v4")
N_ACT = env.action_space.n
N_OB  = env.observation_space.shape
# %%
ob = env.reset()
ac = ActorCritic(N_OB[2],N_ACT).to(DEVICE)

# %%

# %%

def get_action(ob):
    v,p = ac(ob_to_s(ob).to(DEVICE))
    return np.random.choice(N_ACT,p=p.cpu().detach().view(-1).numpy())
def ob_to_s(ob):
    return torch.Tensor(ob.concatenate()).permute(2,0,1).view(-1,N_OB[2],N_OB[0],N_OB[1])

# %%

# %%
%%time
for i in range(2):
    ob = env.reset()
    
    t = 0
    while(1):
        t_start = t
        #env.render()
        act = get_action(ob)
        ob_next,reward,done,info = env.step(act)
        ob = ob_next
        
        t += 1
        if t - 
        
        if done:
            break
        
env.close()
# %%
