import numpy as np
import pandas as pd
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWL
import torchvision
from torchvision import models
import PIL
from PIL import Image
import random
import os
import shutil

class Interpolate(nn.Module):
    def __init__(self, out_sz):
        super().__init__()
        self.out_sz = out_sz

    def go(self, x):
        res = F.interpolate(x, size = (self.out_sz, self.out_sz), mode = "bilinear", align_corners = False)
        assert(len(x.size()) == 4)
        assert(res.size(1) == 3)
        assert(res.size(0) == x.size(0))
        return res

    def forward(self, x):
        if len(x.size()) == 3:
            return self.go(x.unsqueeze(0))[0]
        return self.go(x)
        

class FIDInceptionV3(nn.Module):
    def __init__(
        self,
    ) -> None:
        """
        This class wraps the InceptionV3 model to compute FID.

        Args:
            weights Optional[str]: Defines the pre-trained weights to use.
        """
        super().__init__()
        # pyre-ignore
        self.model = models.inception_v3(weights="DEFAULT")
        # Do not want fc layer
        self.model.fc = nn.Identity()

    def forward(self, x):
        # Interpolating the input image tensors to be of size 299 x 299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.model(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, layers, learning_rate = 1e-3):
        super().__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, betas=(0.5, 0.999))
    
    def forward(self, x):
        if len(x.size()) == 1:
            return self.model(x.unsqueeze(0))[0].squeeze(-1)
        return self.model(x).squeeze(-1)
    
class Classifier(nn.Module):
    def __init__(self, layers, learning_rate = 1e-3):
        super().__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, betas=(0.5, 0.999))
    
    def forward_logits(self, x):
        if len(x.size()) == 3:
            return self.model(x.unsqueeze(0))[0].squeeze()
        return self.model(x).squeeze()
    def forward(self, x):
        logits = self.forward_logits(x)
        probs  = F.softmax(logits, dim = -1)
        return probs
    
    def loss_func(self, x, y):
        logits = self.forward_logits(x)
        loss = F.cross_entropy(logits, y)
        return loss
    def train(self, x, y):
        self.optimizer.zero_grad()
        loss = self.loss_func(x,y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def accuracy(self, x, y):
        probs = self.forward(x)
        preds = torch.argmax(probs, dim = -1)
        correct = torch.sum(preds == y)
        total = len(x)
        return correct, total, correct/total
    
class Generator(nn.Module):
    def __init__(self, layers, latent_size, learning_rate = 1e-3):
        super().__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.latent_size = latent_size
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, betas=(0.5, 0.999))
        
    def forward(self, z):
        if len(z.size()) == 1:
            self.model(z.unsqueeze(0))[0]
        return self.model(z)
    
    def gen_latent_batch(self, B):
        return torch.randn(B, self.latent_size).to(self.DEVICE)
    def generate_batch(self, B):
        z = self.gen_latent_batch(B)
        x = self.forward(z)
        return x
    
class GAN(nn.Module):
    def __init__(self, layers_D, layers_G, latent_size, learning_rate_D = 1e-3, learning_rate_G = 1e-3):
        super().__init__()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers_D = layers_D
        self.layers_G = layers_G
        self.D = Discriminator(layers_D, learning_rate_D)
        self.G = Generator(layers_G, latent_size, learning_rate_G)
    
    def make_batch(self, x):
        sz = len(x.size())
        assert sz >= 1 and sz <= 4
        if sz == 1 or sz == 3:
            return x.unsqueeze(0)
        return x
    
    def generate(self, B):
        return self.G.generate_batch(B)
    def generate_single(self):
        return self.generate(1)[0]
        
    def loss_D(self, x_real):
        B = x_real.size(0)
        x_fake = self.generate(B)
        d_loss_real = BCEWL(self.D(x_real), torch.ones(B).to(self.DEVICE), reduction = 'mean')
        d_loss_fake = BCEWL(self.D(x_fake), torch.zeros(B).to(self.DEVICE), reduction = 'mean')
        d_loss = d_loss_real + d_loss_fake
        return d_loss, x_fake, d_loss_real, d_loss_fake
        
    def loss_G(self, B):
        x = self.generate(B)
        logits = self.D(x)
        g_loss = -torch.mean(F.logsigmoid(logits))
        return g_loss, x
        
    def train_D(self, x_real):
        x_real = self.make_batch(x_real)
        self.D.optimizer.zero_grad()
        loss, x_fake, loss_real, loss_fake = self.loss_D(x_real)
        loss.backward()
        self.D.optimizer.step()
        return loss, x_fake
        
    def train_G(self, B):
        self.G.optimizer.zero_grad()
        loss, x_fake = self.loss_G(B)
        loss.backward()
        self.G.optimizer.step()
        return loss, x_fake
    
    def accuracy_real(self, x_real):
        correct = torch.sum(self.D(x_real) >= 0)
        total   = len(x_real)
        return correct, total, correct/total
        
    def accuracy_fake(self, x_fake):
        correct = torch.sum(self.D(x_fake) < 0)
        total   = len(x_fake)
        return correct, total, correct/total
    
    def accuracy(self, x_real, x_fake = None):
        x_real = self.make_batch(x_real)
        B = x_real.size(0)
        if x_fake is None:
            x_fake = self.generate(B)
        else:
            x_fake = self.make_batch(x_fake)
        correct_real, total_real, acc_real = self.accuracy_real(x_real)
        correct_fake, total_fake, acc_fake = self.accuracy_fake(x_fake)
        assert(total_real == total_fake and total_real == B)
        correct_total = correct_real + correct_fake
        return correct_total, B + B, correct_total/(B + B), correct_real, correct_fake
    





