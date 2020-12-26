# %%
import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.KMNIST("",train=True,download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.KMNIST("",train=False,download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
# %%
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)
classes = datasets.KMNIST.classes
# %% 
# look into the data
for data in trainset:
    #data list[ Tensor 0, Tensor 1]
    print(data[0].shape)
    print(data[1][0].data)
    break

#data_try = data[0].squeeze().permute(0,1,2)
#print(data_try.shape)
# %%
import matplotlib.pyplot as plt
plt.imshow(data[0][0][0])
plt.title(classes[data[1][0]])
plt.show()
# %%
# modify the data
total = 0
counter_dict = {}
for c in classes:
    counter_dict[c]=0
print(counter_dict)
for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[classes[y]]+=1
        total+=1
print(counter_dict) 
# %%
