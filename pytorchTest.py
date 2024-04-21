import torch
from torch.optim import Adam
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torch.nn.functional as F
#the to tensor is what converts the image into numbers, 3 channels, and the brightness 0 to 255 which is scaled to 0 to 1
# normalize is standard deviation and mean normalizing
trainData=datasets.MNIST(root ='data',train=True,transform=ToTensor(),download=True)
testData=datasets.MNIST(root='data',train=False,transform=ToTensor())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(trainData)
print(testData)

print(trainData.data.size())
print(trainData.targets.size())

def GraphSample():
    figure = plt.figure(figsize=(10,8))
    cols,rows =5,5
    for i in range(1,cols*rows+1):
        sample=torch.randint(len(trainData),size=(1,)).item()
        img, label=trainData[sample]
        figure.add_subplot(rows,cols,i)
        plt.title("'Acutal Label: " +str(label))
        plt.axis("off")
        plt.imshow(img.squeeze(),cmap="gray")
    plt.show()
trainLoad=DataLoader(trainData,128,shuffle=True)
testLoad=DataLoader(testData,128,shuffle=False)
classes=('0','1','2','3','4','5','6','7','8','9')

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.conv_relu_stack=nn.Sequential(
                nn.Conv2d(1,32,kernel_size=5),
                nn.ReLU(),
                nn.Conv2d(32,32,kernel_size=5,padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(32,128,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Conv2d(128,128,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(128,256,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Conv2d(256,256,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Flatten()
                )
        self.linear_relu_stack=nn.Sequential(
                nn.Linear(256*3*3,64),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(64,10),
                nn.Softmax(dim=1)
                )
    def forward(self,x):
        x=self.conv_relu_stack(x)
        x=self.linear_relu_stack(x)
        return x

        #Conv,32,5,5 relu
        #same with padding
        #normalization
        #maxpooling 2,2
        #conv2d x2 128,3,3 relu the same steps
        #conv2d 256,3,3 relu x2 repeat same steps
        #flatten
        #dense 64,dropout,10 softmax

model=network()
loss_fn=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),lr=.001,weight_decay=.001)

def testModel():
    model.eval()
    accuracy=0.0
    total=0.0
    with torch.no_grad():
        for data in testLoad:
            images,labels=data
            outputs=model(images)
            _, predicted = torch.max(outputs.data,1)
            total+=labels.size(0)
            accuracy+=(predicted==labels).sum().item()
        accuracy=(100*accuracy/total)
        return(accuracy)
def train(numEpoch):
    bA=0
    for epoch in range(numEpoch):
        runningLoss=0
        runningAcc=0
        for i, (images,labels) in enumerate(trainLoad,0):
            images=Variable(images.to(device))
            labels=Variable(labels.to(device))
            optimizer.zero_grad()
            outputs=model(images)
            loss=loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            runningLoss+=loss.item()
            if i % 1000==999:
                runningLoss=0
            accuracy=testModel()
            if accuracy>bA:
                saveModel()
                bA=accuracy
GraphSample()

