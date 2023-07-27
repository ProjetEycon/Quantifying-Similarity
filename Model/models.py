from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision
import sys,os
from torch import nn
from sentence_transformers import SentenceTransformer
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5,5),padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.unconv1 = nn.ConvTranspose2d(6,3,kernel_size=(5,5),padding=2)
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=(2,2))
        self.unmaxunpool2 = nn.MaxUnpool2d(kernel_size=(2,2))
        self.unmaxunpool3 = nn.MaxUnpool2d(kernel_size=(2,2))

        self.encoder1 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(6, 12,kernel_size=(5,5),padding=2),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(12, 16, kernel_size=(5,5),padding=2),
            nn.Tanh()
        )
        self.encoder3 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(16, 10, kernel_size=(5,5),padding=2),
            nn.Tanh()
        )
        self.encoder4 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(10, 6, kernel_size=(5,5),padding=2),
            nn.Tanh()
        )
        self.decoder4= nn.Sequential(
            nn.ConvTranspose2d(6, 10, kernel_size=(5,5),padding=2),
            nn.Tanh()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(10, 16, kernel_size=(5,5),padding=2),
            nn.Tanh()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 12, kernel_size=(5,5),padding=2),
            nn.Tanh()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(12,6,kernel_size=(5,5),padding=2),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.conv1(x)
        x,indices1 = self.maxpool1(x)
        x = self.encoder1(x)
        x,indices2 = self.maxpool2(x)
        x = self.encoder2(x)
        x,indices3 = self.maxpool3(x)
        x = self.encoder3(x)
        x,indices4 = self.maxpool3(x)
        x = self.encoder4(x)
        shape=x.shape
        z = torch.flatten(x,1)
        x = nn.Unflatten(1, (shape[1], shape[2], shape[3]))(z)
        x = self.decoder4(x)
        x = self.unmaxunpool3(x, indices4)
        x = self.decoder3(x)
        x = self.unmaxunpool3(x, indices3)
        x = self.decoder2(x)
        x = self.unmaxunpool2(x, indices2)
        x = self.decoder1(x)
        x = self.maxunpool1(x,indices1)
        x = self.unconv1(x)
        x = nn.Tanh()(x)
        return z
class transformeur(nn.Module):
        def __init__(self):
            super(transformeur, self).__init__()
            self.model = SentenceTransformer('Sahajtomar/french_semantic')
            if torch.cuda.is_available():
                 self.model=self.model.cuda()
        def forward(self,x):
             embeddings = self.model.encode(x, convert_to_tensor=True)
             return embeddings