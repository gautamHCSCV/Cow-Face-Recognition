#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import time
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import distances
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torchvision
from tqdm.notebook import tqdm
from pytorch_metric_learning import losses


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[4]:


torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# # Data Loading

# In[5]:


config = dict(
    saved_path="saved_models/rough.pt",
    lr=0.001,
    EPOCHS = 5,
    BATCH_SIZE = 16,
    IMAGE_SIZE = 224,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=2,
    USE_AMP = True,
    channels_last=False)

random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])


# In[6]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[7]:


my_path = 'Subset/'
images = torchvision.datasets.ImageFolder(root=my_path,transform=data_transforms['test'])
print(len(images))

images[1][0].shape, images[1][1]


# In[8]:


class Folder_data(Dataset):
    def __init__(self, images, transform = data_transforms):
        super(Folder_data,self).__init__()
        self.train_transforms = transform['test']
        self.test_transforms = transform['test']
        self.is_train = 'True'
        self.to_pil = transforms.ToPILImage()


        self.images = images
        labels = []
        for i in range(len(images)):
            labels.append(images[i][1])
        self.labels = np.array(labels)
        self.index = np.array(list(range(len(self.labels))))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item][0]

        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item][0]
            
            negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
            negative_images = []
            for i in range(50):
                negative_item = random.choice(negative_list)
                negative_img = self.images[negative_item][0]
                negative_images.append(negative_img)

            return anchor_img, positive_img, negative_images, anchor_label
        

test_ds = Folder_data(images)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'])
len(test_loader)


# # Modelling

# In[11]:


def train(model,train_loader,criterion, criterion2,epochs = 10):
    model.train()
    for epoch in range(epochs):
        running_loss = []
        print('Start of Epoch',epoch+1)
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
            anchor_img = anchor_img.to(config['device'])
            positive_img = positive_img.to(config['device'])
            negative_img = negative_img.to(config['device'])
            label = anchor_label.to(config['device'])

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            #loss = criterion(anchor_out, positive_out, negative_out)
            l1,l2 = criterion(anchor_out, positive_out, negative_out), criterion2(anchor_out, label)
            loss =  l1*4+l2*0.2
            #print(l1.cpu().detach().numpy(),l2.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} -Triplet Loss: {:.3f}".format(epoch+1, epochs, np.mean(running_loss)))
    torch.save(model.state_dict(), config['saved_path'])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


# In[12]:


efficientnet = models.efficientnet_b2(pretrained = True)
efficientnet.classifier = Identity()

model = efficientnet
#print(model)
model = model.to(config['device'])

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=2.0)
criterion2 = losses.ArcFaceLoss(50, 1280, margin=28.6, scale=64)


# In[13]:


model.load_state_dict(torch.load('saved_models/eff_b2_1.pt'))
#train(model,train_loader,criterion, criterion2, epochs = 3)


# # Evaluation

# In[15]:


def Angular_evaluation(model,test_loader, threshold = 0.36, num_negative_samples = 50):
    running_loss = []
    cdist = distances.CosineSimilarity()
    correct, total = 0,0
    same_distance, diff_distance = [], []

    model.eval()
    with torch.no_grad():
        for step, (anchor_img, positive_img, negative_images, anchor_label) in enumerate(test_loader):
            anchor_img = anchor_img.to(config['device'])
            positive_img = positive_img.to(config['device'])
            #negative_img = negative_img.to(config['device'])

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
        
            d1 = cdist(anchor_out, positive_out).cpu().detach().numpy()[0]
            same_distance.append(d1)

            if d1>=threshold:
                for i in range(num_negative_samples):
                    negative_img = negative_images[i].to(config['device'])
                    negative_out = model(negative_img)
                    d2 = cdist(anchor_out, negative_out).cpu().detach().numpy()[0]
                    diff_distance.append(d2)
                    
                    if d2>=threshold: # this means that the images are of same class
                        break
                    if i == num_negative_samples-1:
                        correct += 1
                
            total +=1
    print('Correct',correct, ', Total', total, ', Accuracy:', correct/total)
    return(np.array(same_distance), np.array(diff_distance))


# In[16]:


# test_ds = Folder_data(images)
# test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'])

d1, d2 = Angular_evaluation(model,test_loader, threshold = 0.36)
print(sum(d1)/len(d1), sum(d2)/len(d2))
plt.grid()
plt.hist(d1*10, label = 'Same Class Similarity')
plt.hist(d2*10, label = 'Different Class Similarity')
plt.legend()
plt.show()

