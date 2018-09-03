
# coding: utf-8

# In[1]:


#importing some stuff..
import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from reader import *
from utils import *


# In[2]:


#Hyper-Parameters--

#setting the learning rate
lr = 0.0002

#Number of samples to take a random distribution from
rand_num = 9

#Number of epochs
num_epoch = 20

#select min batch size
batchSize = 1
epoch =0
n_epochs = 200

size = 256
input_size = 3
output = 3
decay_epoch = 100


# In[3]:


transforms_ = [ transforms.Resize(int(size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset('datasets/monet2photo', transforms_=transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True, num_workers=4)


# In[4]:


class Residual_block(nn.Module):
    def __init__(self, in_features, use_dropout = True):
        super(Residual_block, self).__init__()
        
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features),
            )
    
    # Connecting the layers
    def forward(self, x):
        return x + self.res(x)
    
class Generator(nn.Module):
    def __init__(self,in_features, output, use_dropout = False, n_block=9):
        super(Generator, self).__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features,64,7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True)
        ]
                    
        n_down = 2
        in_features = 64
        out_features = in_features*2                          #Downsampling
        
        for i in range(n_down):
            f = 2**i
            model += [  nn.Conv2d(in_features*f, out_features*f, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features*f),
                        nn.ReLU(inplace=True) ]
            
        #print out_features*f
        f = 2**(n_down-1)
        
        for i in range(n_block):
            model += [Residual_block(out_features*f)]
        
        in_features = out_features*f
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        #print (str(self.model))
    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_features, out_features=64):
        super(Discriminator, self).__init__()
        
        Dmodel = [
            nn.Conv2d(in_features,out_features,kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace = True)
            ]
        
        in_features = out_features
        for i in range(2):
            f = 2**i
            Dmodel += [
            nn.Conv2d(in_features*f,out_features*f*2,kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_features*f*2),
            nn.LeakyReLU(0.2, inplace = True)
            ]
            
        f = 4    
        Dmodel += [nn.Conv2d(in_features*f, 1, 4, padding=1)]
        
        #Dmodel.append(nn.Sigmoid())

        self.Dmodel = nn.Sequential(*Dmodel)
        print (str(self.Dmodel))
    # Connecting the layers
    def forward(self, x):
        x =  self.Dmodel(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# In[5]:


G_A2B = Generator(input_size, output).cuda()
G_B2A = Generator(output, input_size).cuda()

D_A = Discriminator(input_size).cuda()
D_B = Discriminator(output).cuda()

G_A2B.apply(weights_init_normal)
G_B2A.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

criterion_GAN_M = torch.nn.MSELoss()
criterion_GAN_B = torch.nn.BCELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

print D_B.parameters()
# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor 
input_A = Tensor(batchSize, input_size, size, size)
input_B = Tensor(batchSize, output, size, size)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)



# In[6]:


def backward_D(netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN_M(pred_real, target_real)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN_B(pred_fake, target_fake)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D
    


# In[ ]:


for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        print real_A.size()
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        #idenity loss
        output_B = G_A2B(real_B)
        loss_ID_B = criterion_identity(output_B, real_B)
        
        output_A = G_B2A(real_A)
        loss_ID_A = criterion_identity(output_A, real_A)
        
        identity_loss = (loss_ID_A + loss_ID_B)*5.0
        
        #Gan loss
        fake_B = G_A2B(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN_M(pred_fake, target_real)

        fake_A = G_B2A(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN_M(pred_fake, target_real)

        # Cycle loss
        recovered_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = identity_loss + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        fake_A = fake_A_pool.pop()
        loss_D_A = backward_D(D_A,real_A, fake_A)
        optimizer_D_A.step()
        
        fake_B = fake_B_pool.pop()
        loss_D_B = backward_D(D_B,real_B, fake_B)
        optimizer_D_B.step()
        
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                            images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
        
        

