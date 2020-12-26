#%%
import torch
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])
print(x*y)
# %%
print(x.shape)
# %%
x = torch.zeros([4,3])
y = torch.rand([2,4])
print(x)
print(y)
# %%
import logging
# reshape
try:
    print(y.view([1,8]))
    logging.info("Succesee reshape the pytorch tensor")
except Exception as e:
    pass
print(y.view([4,2]))
try:
    print(y.view([3,3])) # error
except Exception as e:
    logging.error("Fail to reshape the pytorch tensor:"+str(e))
    
# %%
