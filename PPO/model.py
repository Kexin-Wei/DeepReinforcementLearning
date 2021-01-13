# %%
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn as nn

class FC(nn.Module):
    def __init__(self,lr,INPUT_DIM,OUTPUT_DIM,*args):
        super().__init__()
        if args:
            n = len(args[0])
            fc_n = args[0]
        else:
            n = 2
            fc_n = [64,64]
            
        self.n = n
        self.input = nn.Linear(INPUT_DIM,fc_n[0])
        hidden = []
        for i in range(self.n-1):
            hidden.append(nn.Tanh())
            hidden.append(nn.Linear(fc_n[i],fc_n[i+1]))
        self.hidden = nn.Sequential(*hidden)
        self.out = nn.Linear(fc_n[-1],OUTPUT_DIM)
        
        self.device = 'cpu'
        self.to(self.device)
        
        self.optimizer = optim()
    def forward(self,x):
        x = self.out(self.hidden(self.input(x)))
        return x
        
c=FC(8,4,[4,5,6])
print(c)
         
# %%
