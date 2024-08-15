
import torch
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

train_data=datasets.Food101(root="data",split='train',download=True,target_transform=ToTensor())
test_data=datasets.Food101(root="data",split='test',target_transform=ToTensor())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainLoad=DataLoader(train_data,64,shuffle=True)
testLoad=DataLoader(test_data,64,shuffle=False)
#image_shape=train_data[0][0].shape
labels=train_data.classes
print((train_data))
print(labels)
#Try resnet
#Try Inception
#Try EfficientNet
#images are 512x512
class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA,self).__init__()
        self.branch_1=nn.Sequential(
            nn.Conv2d(in,64,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(64,96,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(),
            nn.Conv2d(96,96,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(),
        )
        self.branch_2=nn.Sequential(
            nn.Conv2d(in,48,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(48,64,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(),
        )
        self.branch_3=nn.Sequential(
            nn.AvgPooling(kernel_size=(3,3),stride=(1,1),padding=1),
            nn.Conv2d(in,out,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
        )
        self.branch_4=nn.Sequential(
            nn.Conv2d(in,64,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
        )
    def forward(self,x):
        a=self.branch_1(x)
        b=self.branch_2(x)
        c=self.branch_3(x)
        d=self.branch_4(x)
        out=torch.cat([a,b,c,d],1)
        return out
class Stem(nn.Module):
    def __init__(self):
        super(Stem,self).__init__()
        self.stem_block=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=(3,3),stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=(3,3),stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size(3,3),stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2)),
            nn.Conv2d(64,80,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(80,192,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2)),
        )
    def forward(self,x):
        x=self.stem_block(x)
        return x
class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB,self).__init__()
        self.branch_1=nn.Sequential(
            nn.Conv2d(in,out,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(out,out,kernel_size=(7,1),stride=1,padding=(0,3)),
            nn.ReLU(),
            nn.Conv2d(out,out,kernel_size=(1,7),stride=1,padding=(3,0)),
            nn.ReLU(),
            nn.Conv2d(out,out,kernel_size=(7,1),stride=1,padding=(0,3)),
            nn.ReLU(),
            nn.Conv2d(out,out,kernel_size=(1,7),stride=1,padding=(3,0)),
            nn.ReLU(),
        )
        self.branch_2=nn.Sequential(
            nn.Conv2d(in,out,kernel_size=(1,1),stride=1),
            nn.ReLU(),
            nn.Conv2d(out,out,kernel_size=(1,7),stride=1,padding=(0,3)),
            nn.ReLU(),
            nn.Conv2d(out,192,kernel_size=(7,1),stride=(1,1),padding=(3,0)),
            nn.ReLU(),
        )
        self.branch_3=nn.Sequential(
            nn.AvgPooling(kernel_size=(3,3),stride=(1,1),padding=1),
            nn.Conv2d(in,192,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
        )
        self.branch_4=nn.Sequential(
            nn.Conv2d(in,192,kernel_size=(1,1),stride=(1,1)),
            nn.ReLU(),
        )
        def forward(self,x):
            a=branch_4(x)
            b=branch_2(x)
            c=branch_1(x)
            d=branch_3(x)
            out=torch.cat([a,b,c,d],1)
            return out
Class ReductionA(nn.Module):
def __init__(self):
    super(ReductionA,self).__init__()
    self.branch_1=nn.Sequential(
        nn.Conv2d(in,64,kernel_size=(1,1),stride=(1,1)),
        nn.ReLU(),
        nn.Conv2d(64,96,kernel_size=(3,3),stride=(1,1),padding=1),
        nn.ReLU(),
        nn.Conv2d(96,96,kernel_size=(3,3),stride=2),
        nn.ReLU(),
    )
    self.branch_2=nn.Sequential(
        nn.Conv2d(in,384,kernel_size=(3,3),stride=(2,2)),
        nn.ReLU(),
    )
    self.branch_3=nn.Sequential(
        nn.MaxPool2d(kernel_size=(3,3),stride=2)
    )
    def forward(self,x):
        a=self.branch_1(x)
        b=self.branch_2(x)
        c=self.branch_3(x)
        out=torch.cat([a,b,c],1)
        return out
class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB,self).__init__()
        self.branch_1=nn.Sequential(
            nn.Conv2d(in,192,kernel_size=(1,1),stride=1),
            nn.ReLU(),
            nn.Conv2d(192,192,kernel_size=(1,7),stride=1,padding=(0,3)),
            nn.ReLU(),
            nn.Conv2d(192,192,kernel_size=(7,1),stride=1,padding=(3,0)),
            nn.ReLU(),
            nn.Conv2d(192,192,kernel_size=(3,3),stride=2),
            nn.ReLU(),
        )
        self.branch_2=nn.Sequential(
            nn.Conv2d(in,192,kernel_size=(1,1),stride=1),
            nn.ReLU(),
            nn.Conv2d(192,320,kernel_size=(1,1),stride=2),
            nn.ReLU(),
        )
        self.branch_3=nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
        )
        def forward(self,x):
            a=self.branch_1(x)
            b=self.branch_2(x)
            c=self.branch_3(x)
            out=torch.cat([a,b,c],1)
            return out
class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.InceptionA1=InceptionA(192,32)
        self.InceptionA2=InceptionA(288,64)
        self.InceptionA3=InceptionA(288,64)
        self.reduceA=ReductionA(288)
model=Network()
loss=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),lr=.001,weight_decay=.001)

def test():
    model.eval()
    acc=0.0
    total=0.0
    with torch.no_grad():
        for data in testLoad:
            images,labels=data
            outputs=model(images)
            _,pred=torch.max(outputs.data,1)
            total+=labels.size(0)
            acc+=(pred==labels).sum().item()
        acc=(100*acc/total)
        return acc
def train(epoch):
    for e in range(epoch):
        runningLoss=0
        runningacc=0
        for i,(images,labels) in enumerate(trainLoad,0):
        

        #input,stem3incepA*3,reducA,incepB*4,reducB,incepC*2,avgpool
