# KMNIST torch Fully Connected Neural Network classifier

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torchvision import transforms, datasets
# %%
train = datasets.KMNIST("",train=True,download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.KMNIST("",train=False,download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
# %%
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)
classes = datasets.KMNIST.classes
# %%
data_batch_0=list(trainset)[0][0]
INPUT_SIZE = data_batch_0.shape[2]*data_batch_0.shape[3]
OUTPUT_SIZE = len(classes)
print(INPUT_SIZE,OUTPUT_SIZE)
# %%
## Build Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(INPUT_SIZE,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,OUTPUT_SIZE)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x),dim=1)
        return x
    
# %%

# %%
net = Net()
print(net)
# %%
y = net(data_batch_0.view(-1,INPUT_SIZE))
print(data_batch_0.view(-1,INPUT_SIZE).shape)
print(y.shape)
# %%
optimizer = optim.Adam(net.parameters(),lr=0.001)

EPOCHS = 3
# %%
for ep in range(EPOCHS):
    for data in trainset:
        feature,label = data
        
        net.zero_grad()
        predict = net(feature.view(-1,INPUT_SIZE))
        loss = F.nll_loss(predict,label)
        loss.backward()
        optimizer.step()
    print(loss)
# %%
correct = 0
total   = 0
with torch.no_grad():
    for data in testset:
        feature, label = data
        predict = net(feature.view(-1,INPUT_SIZE))
        for index, i in enumerate(predict):
            if torch.argmax(i) == label[index]:
                correct +=1
            total +=1
        
print("Accuracy:",correct/total)
# %%
print(correct)
# %%
print(torch.argmax(predict[0]))
print(label[0])

import matplotlib.pyplot as plt
plt.imshow(feature[0][0])
title = f"Predict:{classes[torch.argmax(predict[0])]}, Label:{classes[label[0]]}"
plt.title(title)
plt.show()
# %%
