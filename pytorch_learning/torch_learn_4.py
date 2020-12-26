# torch 1.7.1
# dog cat cnn classifier

# %%
import os
import cv2
import random

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from sys import platform
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

# %%
print(platform)
# %%
ROOT = os.path.expanduser('~')
DIR  = os.path.join(ROOT,"Downloads/dogs-vs-cats-redux-kernels-edition")

TRAIN_DIR = os.path.join(DIR,'train')
TEST_DIR = os.path.join(DIR,'test')
classes = ["cat","dog"]
IMAGE_SIZE = 80
# %%
"""
for f in os.listdir(TRAIN_DIR):
    image = cv2.imread(os.path.join(TRAIN_DIR,f),cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_AREA)
    label = 0 if f.startswith("cat") else 1
    plt.imshow(image)
    plt.axis('off')
    plt.title(classes[label])
    plt.show()
    break
"""
# %%
data = []
cat_num = 0
dog_num = 0
for f in os.listdir(TRAIN_DIR):
    image = cv2.imread(os.path.join(TRAIN_DIR,f),cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_AREA)
    label = 0 if f.startswith(classes[0]) else 1
    if label == 0: cat_num+=1 
    else: dog_num+=1
    data.append([image,label])
    
print(len(data),cat_num,dog_num)
# %%

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d( 1,  32, 5)
        self.conv2 = nn.Conv2d(32,  64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(IMAGE_SIZE,IMAGE_SIZE).view(-1,1,IMAGE_SIZE,IMAGE_SIZE)
        self._to_linear = None
        
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,2)
    
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2,2)
        x = F.max_pool2d(F.relu(self.conv3(x)),2,2)
        
        if self._to_linear is None:
            shape =  x[0].shape
            print(shape)
            self._to_linear =shape[0]*shape[1]*shape[2] 
        return x
    
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

# %%
import torch.optim as optim

optimizer = optim.Adam(net.parameters(),lr=0.001)

loss_function = nn.NLLLoss()

# %%
def split(data,ratio):
    l = len(data)
    random.shuffle(data)
    split_index = int(l*(1-ratio))
    train = data[:split_index]
    valid = data[split_index:]
    
    train_feature, train_label = [],[]
    valid_feature, valid_label = [],[]
    for i in train:
        train_feature.append(i[0])
        train_label.append(i[1])
    for i in valid:
        valid_feature.append(i[0])
        valid_label.append(i[1])
    
    train_feature = torch.Tensor(train_feature)/255.0
    train_label   = torch.Tensor(train_label).type(torch.LongTensor)
    valid_feature = torch.Tensor(valid_feature)/255.0
    valid_label   = torch.Tensor(valid_label).type(torch.LongTensor)
    return (train_feature,train_label),(valid_feature,valid_label)

train,valid = split(data,0.2)
#print(len(train[1]),len(valid[1]))

# %%
net = CNN()
print(net)
BATCH_SIZE = 100
EPOCHS = 10
# %%
for ep in range(EPOCHS):
    loss_sum = 0

    prefix =f"Epoch: {ep}" 
    
    tqdm_bar = tqdm(range(0,len(train[1]),BATCH_SIZE),desc=prefix)
    for i in tqdm_bar:
        batch_feature = train[0][i:i+BATCH_SIZE].view(-1,1,IMAGE_SIZE,IMAGE_SIZE)
        batch_label   = train[1][i:i+BATCH_SIZE]
        
        net.zero_grad()
        
        predict = net(batch_feature)
        
        loss = loss_function(predict,batch_label)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
    tqdm_bar.set_postfix_str(f"loss:{loss_sum}")
# %%
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(valid[1])):
        valid_predict = torch.argmax(net(valid[0][i].view(-1,1,IMAGE_SIZE,IMAGE_SIZE)))
        if valid_predict == valid[1][i]:
            correct +=1 
        total+=1   
        
accuracy = round(correct/total,3)
print(accuracy)    
# %%
plt.imshow(valid[0][2])
plt.axis('off')
plt.title(classes[valid[0][2]])
plt.show()
# %%
