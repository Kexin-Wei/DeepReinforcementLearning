# %%
import torch
import torch.nn as nn
import torch.nn.functional as F 

#%%
class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn1 = nn.Conv2d(1,32,8,stride=4)
        self.cnn2 = nn.Conv2d(32,64,4,stride=2)     
        self.cnn3 = nn.Conv2d(64,64,3,stride=1)
        self.value  = nn.Linear()