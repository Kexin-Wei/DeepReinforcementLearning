import numpy as np
class Replay:
    def __init__(self, BATCH_SIZE = 32):
        
        self.BATCH_SIZE = BATCH_SIZE
        
        self.obs  = []
        self.obs_ = []
        self.acts = []
        self.dones   = []
        self.rewards = []
        self.advantages = []
        
    @property
    def len(self):
        return len(self.obs)
    
    def append(self,obs,obs_,acts,dones,rewards,advantages):
        self.obs.extend(obs)
        self.obs_.extend(obs_)
        self.acts.extend(acts)
        self.dones.extend(dones)
        self.rewards.extend(rewards)
        self.advantages.extend(advantages)
    
    def sample(self):
        indexs = np.random.choice(range(self.len),
                    self.BATCH_SIZE,replace=False)
        obs  = []
        obs_ = []
        acts = []
        dones   = []
        rewards = []
        advantages = []
        for idx in indexs:
            obs.append(self.obs[idx])
            obs_.append(self.obs_[idx])
            acts.append(self.acts[idx])
            dones.append(self.dones[idx])
            rewards.append(self.rewards[idx])
            advantages.append(self.advantages[idx])
        
        return obs,obs_,acts,dones,rewards,advantages
    
    def return_all(self):
        return self.obs, self.obs_, self.acts,\
               self.dones,self.rewards,self.advantages