from model import AC_FC
import torch

class Agent():
    def __init__(self, ALPHA, IN_DIMS, N_ACT, GAMMA = 0.99, 
                 FC1_DIMS = 256, FC2_DIMS = 256):
        self.GAMMA = GAMMA
        
        self.ac = AC_FC(ALPHA, IN_DIMS, N_ACT,  FC1_DIMS = FC1_DIMS, FC2_DIMS = FC2_DIMS)
    
    def get_action(self, ob):    
        v, p = self.ac.forward(ob)
        dist = torch.distributions.Categorical(p) # discrete distribution with N_ACTs probabilities
        act  = dist.sample()
        return act.item()
    
    def train(self, ob, act, reward, next_ob, done):
        
        self.ac.optimizer.zero_grad()
        
        v,p = self.ac.forward(ob)
        next_v , next_p = self.ac.forward(next_ob)
        
        reward = torch.Tensor([reward]).to(self.ac.device)
                
        # TD error
        td_error  = reward + (1-done)*self.GAMMA* next_v - v
        
        # actor loss <- negative *log(p(ob,act)*TD error
        actor_loss  = - p.log()[act] * td_error
        critic_loss = td_error**2
        
        (actor_loss+critic_loss).backward()
        self.ac.optimizer.step()
        
        
                 