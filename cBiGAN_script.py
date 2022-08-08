import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms

from hdf5_dataset import HDF5Dataset

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define some global variables
nz = 100
batch_size = 512
beta1 = 0.5
beta2 = 0.999
n_epochs = 400
l_rate = 2e-5

# dataset preparation
def load_dataset(batch_size = batch_size, path = "/datax/scratch/zelakiewicz/"):
    dataset = HDF5Dataset(file_path=path , recursive=False, load_data=True, transform=transforms.ToTensor())

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
   
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128 * 128, 1024),
            nn.LeakyReLU(0.2),
            # nn.Linear(2048,1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, nz)
        )
        
    def forward(self, X):
        return self.layers(X)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(nz + 6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 2048),
            # nn.BatchNorm1d(2048),
            nn.Linear(1024, 128 * 128),
            nn.Sigmoid()
        )

        self.layers_leaky = nn.Sequential(
            nn.Linear(nz + 6, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(1024, 2048),
            # nn.BatchNorm1d(2048),
            nn.Linear(1024, 128 * 128),
            nn.Sigmoid()
        )
        
    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.layers(zc)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128 * 128 + nz + 6, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.Dropout(0.4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            # nn.Linear(1024,2048),
            # nn.Dropout(0.4),
            # nn.BatchNorm1d(2048),
            # nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        
    def forward(self, X, z, c):
        Xzc = torch.cat([X, z, c], dim=1)
        return self.layers(Xzc)

def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
    return -torch.mean(loss)

def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

E = Encoder().to(device)
G = Generator().to(device)
D = Discriminator().to(device)

E.apply(init_weights)
G.apply(init_weights)
D.apply(init_weights)

#optimizers with weight decay
optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), 
                                lr=l_rate, betas=(beta1, beta2), weight_decay=1e-5)
optimizer_D = torch.optim.Adam(D.parameters(), 
                               lr=l_rate, betas=(beta1, beta2), weight_decay=1e-5)

mnist_train, mnist_test = load_dataset()

lossD_list = []
accD_list = []
lossEG_list = []
accEG_list = []
epoch_list = []

for epoch in range(n_epochs):
    D_loss_acc = 0.
    EG_loss_acc = 0.
    D.train()
    E.train()
    G.train()
        
#     scheduler_D.step()
#     scheduler_EG.step()
    
    for i, (images, labels) in enumerate(tqdm(mnist_train)):
        images = images.to(device)
        images = images.reshape(images.size(0),-1)
        images = F.normalize(images, dim=1)

        #make one-hot embedding from labels
        c = torch.zeros(images.size(0), 6, dtype=torch.float32).to(device)
        c[torch.arange(images.size(0)), labels] = 1
        
        #initialize z from 100-dim U[-1,1]
        z = torch.rand(images.size(0), nz)
        z = z.to(device)
        
        # Start with Discriminator Training
        optimizer_D.zero_grad(set_to_none=True)
    
        #compute G(z, c) and E(X)
        Gz = G(z, c)
        EX = E(images)
        
        #compute D(G(z, c), z, c) and D(X, E(X), c)
        DG = D(Gz, z, c)
        DE = D(images, EX, c)
        
        #compute losses
        loss_D = D_loss(DG, DE)
        D_loss_acc += loss_D.item()
        
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        #Encoder & Generator training
        optimizer_EG.zero_grad(set_to_none=True)
        
        #compute G(z, c) and E(X)
        Gz = G(z, c)
        EX = E(images)
        
        #compute D(G(z, c), z, c) and D(X, E(X), c)
        DG = D(Gz, z, c)
        DE = D(images, EX, c)
        
        #compute losses
        loss_EG = EG_loss(DG, DE)
        EG_loss_acc += loss_EG.item()

        loss_EG.backward()
        optimizer_EG.step()


    lossD_list.append(D_loss_acc / i)
    accD_list.append(D_loss_acc)
    lossEG_list.append(EG_loss_acc / i)
    accEG_list.append(EG_loss_acc)
    epoch_list.append(epoch+1)

    if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
        # print('Epoch [{}/{}], Avg_Loss_D: {:.4f}, Avg_Loss_EG: {:.4f}'
            #   .format(epoch + 1, n_epochs, D_loss_acc / i, EG_loss_acc / i))
        n_show = 10
        D.eval()
        E.eval()
        G.eval()
        
        with torch.no_grad():
            #generate images from same class as real ones
            real = images[:n_show]
            c = torch.zeros(n_show, 6, dtype=torch.float32).to(device)
            c[torch.arange(n_show), labels[:n_show]] = 1
            z = torch.rand(n_show, nz)
            z = z.to(device)
            gener = G(z, c).reshape(n_show, 128, 128).cpu().numpy()
            recon = G(E(real), c).reshape(n_show, 128, 128).cpu().numpy()
            real = real.reshape(n_show, 128, 128).cpu().numpy()

            fig, ax = plt.subplots(3, n_show, figsize=(15, 6))
            fig.subplots_adjust(wspace=0.05, hspace=0)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('Epoch {}'.format(epoch+1))
            fig.text(0.04, 0.75, 'G(z, c)', ha='left')
            fig.text(0.04, 0.5, 'x', ha='left')
            fig.text(0.04, 0.25, 'G(E(x), c)', ha='left')
            fig.text(0.5, 0.05, f'LR: {l_rate}    B1: {beta1}    B2: {beta2}    BatchSize: {batch_size}', ha='center')

            for i in range(n_show):
                ax[0, i].imshow(gener[i], cmap='gray')
                ax[0, i].axis('off')
                ax[1, i].imshow(real[i], cmap='gray')
                ax[1, i].axis('off')
                ax[2, i].imshow(recon[i], cmap='gray')
                ax[2, i].axis('off')
            plt.savefig('figs/linear2/epoch_'+str(epoch+1)+'_ol.jpg')
            plt.clf()
            
    
    if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 :

        print('Epoch [{}/{}], Avg_Loss_D: {:.4f}, Avg_Loss_EG: {:.4f}, Acc_EG: {:.4f}, Acc_D: {:.4f}'
              .format(epoch + 1, n_epochs, D_loss_acc / i, EG_loss_acc / i, EG_loss_acc, D_loss_acc))
        fig, ax = plt.subplots(figsize=(15, 10))
        # ax2 = ax.twinx()
        plt.title("Average loss")# and Accuracy")

        d_loss = ax.plot(epoch_list, lossD_list, label="Discriminator Loss", color='mediumorchid')
        eg_loss = ax.plot(epoch_list, lossEG_list, label="Generator & Encoder Loss", color = 'orange')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        # d_acc = ax2.plot(epoch_list, accD_list, label="Discriminator Accuracy", color='skyblue')
        # eg_acc = ax2.plot(epoch_list, accEG_list, label="Generator & Encoder Accuracy", color='salmon')
        # ax2.set_ylabel("Accuracy")
        
        plots = d_loss + eg_loss #+ d_acc + eg_acc
        labels = [l.get_label() for l in plots]
        ax.legend(plots, labels, loc='upper right')

        ax.grid()
        # ax2.grid()

        plt.tight_layout()
        plt.savefig('figs/linear2/epoch_'+str(epoch+1)+'_loss.jpg')
        plt.clf()

# torch.save({
            # 'D_state_dict': D.state_dict(),
            # 'E_state_dict': E.state_dict(),
            # 'G_state_dict': G.state_dict(),
            # 'optimizer_D_state_dict': optimizer_D.state_dict(),
            # 'optimizer_EG_state_dict': optimizer_EG.state_dict(),
            # 'scheduler_D_state_dict': scheduler_D.state_dict(),
            # 'scheduler_EG_state_dict': scheduler_EG.state_dict()
            # }, '.\models\models_state_dict_CBiGAN.tar')
            