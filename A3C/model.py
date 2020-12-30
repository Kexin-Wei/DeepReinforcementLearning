# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class ActorCritic(nn.Module):
    def __init__(self,IN_CHANNELS,N_ACT):
        super().__init__()
        
        self.conv1 = nn.Conv2d(IN_CHANNELS,16,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc    = nn.Linear(9*9*32,256)
        
        self.value  = nn.Linear(256,1)
        self.policy = nn.Linear(256,N_ACT)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc(x))
        
        v = self.value(x)
        p = F.softmax(self.policy(x))
        return v,p
# %%
# from dm_wrapper import make_env
# env = make_env("PongNoFrameskip-v4")
# N_ACT = env.action_space.n
# N_OB  = env.observation_space.shape
# # %%
# ob = env.reset()
# ac = ActorCritic(N_OB[2],N_ACT)

# def ob_to_s(ob):
#     return torch.Tensor(ob.concatenate()).permute(2,0,1).view(-1,N_OB[2],N_OB[0],N_OB[1])
# #%%
# for i in range(100):
#     env.render()
#     act = env.action_space.sample()
#     ob_next,reward,done,info = env.step(act)
#     v, p = ac(ob_to_s(ob))
#     ob = ob_next
#     if done :
#         break
# %%
