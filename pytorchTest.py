import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

#the to tensor is what converts the image into numbers, 3 channels, and the brightness 0 to 255 which is scaled to 0 to 1
# normalize is standard deviation and mean normalizing
trainData=datasets.MNIST(root ='data',train=True,transform=ToTensor(),download=True)
testData=datasets.MNIST(root='data',train=False,transform=ToTensor())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
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

GraphSample()
