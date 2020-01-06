#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt

# plot p0 and p1
#plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
#plt.hist(f(xx), 100, alpha=0.5, density=1)
#plt.hist(xx, 100, alpha=0.5, density=1)
#plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
#plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
#plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch import utils

import samplers as sp

# Dicriminator (MLP):

class Discriminator(nn.Module):
    """Dicriminator"""        
    
    def __init__(self, input_size = 512):
      
        super(Discriminator, self).__init__()
        self.input_size = input_size
        
        self.mlp = nn.Sequential(
            # Linear layer 1
            nn.Linear(input_size, 512),
            nn.ReLU(),
            
            # Linear layer 2
            nn.Linear(512, 512),
            nn.ReLU(),
            
            # Linear layer 3
            nn.Linear(512, 1),
            #nn.Sigmoid() # not if using BCEwithlogitloss
        )  
    
    def forward(self, x):
        return self.mlp(x)

batch_size = 512

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# instanciate a Discriminator:
d = Discriminator(batch_size).to(device)

# training:
    
num_epochs = 100 # number of training epochs
init_lr = 0.001 # initial learning rate

# the binary cross entropy l(y,D(x)) is: -1 * [y log(D(x)) + (1 - y)*log(1 - D(x))]
# if we take y to be zero (distribution q), we optimize -log(1-D(x))
# if we take y to be 1 (distribution p), we optimize -log(D(x))
# we want to minimize this (this is directly equivalent to maximizing our objective function)
criterion = nn.BCEWithLogitsLoss() # binary cross entropy with built-in sigmoid
optimizer = optim.SGD(d.parameters(), lr=init_lr)

for epoch in range(num_epochs):
    
    # sample minibatches from the two distributions:
    distr1 = sp.distribution1(0, batch_size)
    dist1 = iter(distr1)
    samples1 = np.squeeze(next(dist1)[:,0])
    t1 = torch.Tensor(samples1).to(device)
    distr3 = sp.distribution3(batch_size)
    dist3 = iter(distr3)
    samples3 = np.squeeze(next(dist3))
    t3 = torch.Tensor(samples3).to(device)
    
    d.zero_grad() # gradients to zero
    
    # gradients on 'real' distribution:
    out_r = d(t1)
    err_r = criterion(out_r, torch.Tensor([1]).to(device))
    err_r.backward()
    
    # gradients on 'fake' distribution:
    out_f = d(t3)
    err_f = criterion(out_f, torch.Tensor([0]).to(device))
    err_f.backward()
    
    optimizer.step() # this maximizes our objective function (minimizes -1 * objective)
    
    JSdiv = out_f.mean().item()
    print('End of epoch ', epoch,', total error : ', (err_r + err_f).item())
    #D_x = out_f.mean().item()
    #D_G_z1 = output.mean().item()
    
    # Add the gradients from the all-real and all-fake batches
    #errD = errD_real + errD_fake
  

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density



r = xx # evaluate xx using your discriminator; replace xx with the output
#plt.figure(figsize=(8,4))
#plt.subplot(1,2,1)
#plt.plot(xx,r)
#plt.title(r'$D(x)$')

estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0." with your estimate
#plt.subplot(1,2,2)
#plt.plot(xx,estimate)
#plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
#plt.legend(['Estimated','True'])
#plt.title('Estimated vs True')

#plt.show();









