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
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item][0]

            if self.train_transforms:
                anchor_img = self.train_transforms(self.to_pil(anchor_img))
                positive_img = self.train_transforms(self.to_pil(positive_img))
                negative_img = self.train_transforms(self.to_pil(negative_img))

                return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.test_transforms(self.to_pil(anchor_img))
            return anchor_img


# In[9]:


import pickle
a_file = open("dataset_50.pkl", "rb")
dataset = pickle.load(a_file)
print(len(set(dataset['labels'])), len(dataset['labels']))

img = np.transpose(dataset['images'][0],(1,2,0))
print(img.shape)
plt.imshow(img)
plt.show()


# In[10]:


class Custom_data(Dataset):
    def __init__(self, dataset, n_class = 50, transform = data_transforms, train=True):
        super(Custom_data,self).__init__()
        self.train_transforms = transform['train']
        self.test_transforms = transform['test']
        self.is_train = train
        self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.images = dataset['images']
            self.labels = np.array(dataset['labels'])
            self.index = np.array(list(range(len(self.labels))))
            self.n_class = n_class

        else:
            self.images = dataset['images']
            self.n_class = 1

    def __len__(self):
        return len(self.images)*(self.n_class-1)

    def __getitem__(self, item):
        item = item//(self.n_class-1)
        anchor_img = self.images[item]

        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item]

            negative_label = item % (self.n_class-1)
            if negative_label>=anchor_label: negative_label+=1
            negative_list = self.index[self.index!=item][self.labels[self.index!=item]==negative_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item]
            if anchor_label==negative_label: print('Error')

            if self.train_transforms:
                anchor_img = self.train_transforms(self.to_pil(anchor_img))
                positive_img = self.train_transforms(self.to_pil(positive_img))
                negative_img = self.train_transforms(self.to_pil(negative_img))

                return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.test_transforms(self.to_pil(anchor_img))
            return anchor_img


# In[11]:


train_ds = Custom_data(dataset, train=True)
train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['num_workers'])
valid_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=config['num_workers'])

a = iter(valid_loader)
b = next(a)
print(b[0].shape, b[1].shape, b[2].shape, b[3].shape)

plt.imshow(b[0][0][2])
plt.show()

len(train_loader)

# test_ds = Folder_data(images)
# test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'])


# # Modelling

# In[12]:


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


# In[13]:


efficientnet = models.efficientnet_b0(pretrained = True)
efficientnet.classifier = Identity()

model = efficientnet
#print(model)
model = model.to(config['device'])

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=2.0)
criterion2 = losses.ArcFaceLoss(50, 1280, margin=28.6, scale=64)


# In[14]:


model.load_state_dict(torch.load('saved_models/rough.pt'))
train(model,train_loader,criterion, criterion2, epochs = 3)


# # Evaluation

# In[15]:


def Angular_evaluation(model,test_loader, threshold = 0.5):
    running_loss = []
    pdist = distances.CosineSimilarity()
    scorrect, dcorrect, total = 0,0,0
    same_distance, diff_distance = [], []

    model.eval()
    with torch.no_grad():
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(test_loader):
            anchor_img = anchor_img.to(config['device'])
            positive_img = positive_img.to(config['device'])
            negative_img = negative_img.to(config['device'])

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            d1 = pdist(anchor_out, positive_out).cpu().detach().numpy()[0]
            d2 = pdist(anchor_out, negative_out).cpu().detach().numpy()[0]
            same_distance.append(d1)
            diff_distance.append(d2)

            if d1>=threshold: scorrect += 1
            if d2<threshold: dcorrect += 1
#                 image1 = anchor_img.cpu().detach().numpy()[0]
#                 image2 = negative_img.cpu().detach().numpy()[0]
                
#                 fig, (ax1, ax2) = plt.subplots(1, 2)
#                 ax1.axis('off')
#                 ax1.imshow(np.transpose(image1,(1,2,0)))
#                 ax2.axis('off')
#                 ax2.imshow(np.transpose(image2,(1,2,0)))
#                 plt.show()

                
            total +=1
    print('Correct in same class',scorrect, ', Total', total, ', Accuracy:', scorrect/total)
    print('Correct in different class', dcorrect, ', Accuracy:', dcorrect/total)
    return(np.array(same_distance), np.array(diff_distance))


# In[17]:


test_ds = Folder_data(images)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'])

d1, d2 = Angular_evaluation(model,test_loader, threshold = 0.22)
print(sum(d1)/len(d1), sum(d2)/len(d2))
plt.grid()
plt.hist(d1*10, label = 'Same Class Similarity')
plt.hist(d2*10, label = 'Different Class Similarity')
plt.legend()
plt.show()


# In[27]:


def Evaluate(model,test_loader, threshold = 2.4):
    running_loss = []
    pdist = torch.nn.PairwiseDistance(p=2)
    cdist = distances.CosineSimilarity()
    scorrect, dcorrect, total = 0,0,0
    same_distance, diff_distance = [], []
    sang_distance, dang_distance = [], []

    model.eval()
    with torch.no_grad():
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(test_loader):
            anchor_img = anchor_img.to(config['device'])
            positive_img = positive_img.to(config['device'])
            negative_img = negative_img.to(config['device'])

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            d1 = pdist(anchor_out, positive_out).cpu().detach().numpy()[0]
            ang1 = cdist(anchor_out, positive_out).cpu().detach().numpy()[0]
            d2 = pdist(anchor_out, negative_out).cpu().detach().numpy()[0]
            ang2 = cdist(anchor_out, negative_out).cpu().detach().numpy()[0]

            same_distance.append(d1)
            diff_distance.append(d2)
            sang_distance.append(ang1)
            dang_distance.append(ang2)

            if d1<=threshold or ang1>0.4: scorrect += 1
            if d2>threshold or ang2<0.25: dcorrect += 1
            total +=1
    print('Correct in same class',scorrect, ', Total', total, ', Accuracy:', scorrect/total)
    print('Correct in different class', dcorrect, ', Accuracy:', dcorrect/total)
    print(np.mean(np.array(sang_distance)), np.mean(np.array(dang_distance)))
    return(same_distance, diff_distance)


# In[28]:


d1, d2 = Evaluate(model,test_loader, threshold = 7.0)


# In[29]:


print(sum(d1)/len(d1), sum(d2)/len(d2))
plt.grid()
plt.hist(d1, label = 'Same Class Distance')
plt.hist(d2, label = 'Different Class Distance')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:


device = 'cpu'
model.to(device)
print()


# In[ ]:


def evaluate(threshold = 2.5):
    pdist = torch.nn.PairwiseDistance(p=2)
    correct = total = 0
    same_class = []
    different_class = []

    model.eval()
    for i in range(25,min(len(images),80)):
        x1 = images[i][0].unsqueeze(0).to(device)
        y1 = model(x1)

        for j in range(i+1, min(len(images), i+6)):
            x2 = images[j][0].unsqueeze(0).to(device)
            y2 = model(x2)

            d = pdist (y1, y2)[0]
            if d<=threshold and images[i][1]==images[j][1]:
                correct += 1
            if d>threshold and images[i][1]!=images[j][1]: correct += 1

            if images[i][1]==images[j][1]:
                same_class.append(d)
            else:
                different_class.append(d)
            total+=1
        #torch.cuda.empty_cache()

    print('Correct',correct, '\n', 'Total', total)
    print(correct/total)
    return (sum(same_class)/len(same_class), sum(different_class)/len(different_class))

evaluate(threshold = 2.8)


# In[ ]:


def angular_evaluation(threshold = 0.90):
    pdist = distances.CosineSimilarity()
    correct = total = 0
    same_class = []
    different_class = []

    model.eval()
    for i in range(45):
        x1 = images[i][0].unsqueeze(0).to(device)
        y1 = model(x1)
        for j in range(i+1, min(len(images), i+8)):
            x2 = images[j][0].unsqueeze(0).to(device)
            y2 = model(x2)

            d = pdist (y1, y2)[0]
            if d>threshold and images[i][1]==images[j][1]:
                correct += 1
            if d<=threshold and images[i][1]!=images[j][1]: correct += 1

            if images[i][1]==images[j][1]:
                same_class.append(d)
            else:
                different_class.append(d)
            total+=1
        #torch.cuda.empty_cache()

    print('Correct',correct, '\n', 'Total', total)
    print(correct/total)
    return (sum(same_class)/len(same_class), sum(different_class)/len(different_class))

angular_evaluation(threshold = 0.50)


# In[ ]:





# In[ ]:




