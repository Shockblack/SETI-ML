
# Importing all necessary packages
#Starting with PyTorch and their ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter

# Other usefull packages
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# My custom h5 dataset
from hdf5_dataset import HDF5Dataset

# Running on GPU if possible
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Tensorboard setup
# tb = SummaryWriter()

# Setting a bunch of global variables for various purposes
workers = 8 # number of workers for cpu parallelization
batch_size = 128 # batch size
lr = 2e-5 # learning rate
n_epochs = 400 # number of epochs

# Size of image, amount of feature maps, latent vector size, and number of channels
img_size = 128 # image size in pixels
ngf = 128 # number of generator features
ndf = 128 # number of discriminator features
nef = 128 # number of encoder features
nz = 100 # latent vector size
nc = 1 # number of channels
nf = 6 # number of features

# Creating data loaders for training and testing sets
def load_dataset(batch_size = batch_size, path = "/datax/scratch/zelakiewicz/"):
    dataset = HDF5Dataset(file_path=path , recursive=False, load_data=True, transform=transforms.ToTensor())

    train_dataset, test_dataset = random_split(dataset, [25000, 5000])
   
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader

# Initialize custom weights
def init_weights(Layer):
    name = Layer.__class__.__name__
    # Using a mean of 0 and std of 0.2 as per standard
    if name.find('Conv') != -1:
        nn.init.normal_(Layer.weight.data, mean=0.0, std=0.02)

    elif name.find('BatchNorm') != -1:
        nn.init.normal_(Layer.weight.data, mean=1.0, std=0.2)
        nn.init.constant_(Layer.bias.data, 0)

# This is a plug-in-play replacement to nn.Unflatten
# Thanks to GitHub user @jungerm2
class View(nn.Module):
    def __init__(self, dim,  shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)

# Creating the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.generator_lin = nn.Sequential(
            # input is [nz + nf] = [100 + 6]
            nn.Linear(nz + nf, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 8 * 8 * ngf * 8),
        )

        self.unflatten = View(dim=1, shape=(ngf*8, 8, 8)) # out: 1024 x 8 x 8

        self.flatten = nn.Flatten(start_dim=1)

        self.generator_conv = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False), # out: 512 x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False), # out: 256 x 32 x 32
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False), # out: 128 x 64 x 64
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf, 1, kernel_size=4, stride=2, padding=1, bias=False), # out: 1 x 128 x 128
            nn.Sigmoid()
        )


    def forward(self, z, c):
        # Concatenate inputs
        zc = torch.cat([z, c], dim=1)

        zc = self.generator_lin(zc)
        zc = self.unflatten(zc)
        zc = self.generator_conv(zc)
        zc = self.flatten(zc)
        return zc

# Creating the Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Convolution Section
        self.encoder_conv = nn.Sequential(
            # input is [nc, img_len, img_len] = [1, 128, 128]
            nn.Conv2d(nc, nef, kernel_size=4, stride=2, padding=1, bias=False), # out: 128 x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef, nef*2, kernel_size=4, stride=2, padding=1, bias=False), # out: 256 x 32 x 32
            nn.BatchNorm2d(nef*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef*2, nef*4, kernel_size=4, stride=2, padding=1, bias=False), # out: 512 x 16 x 16
            nn.BatchNorm2d(nef*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nef*4, nef*8, kernel_size=4, stride=2, padding=1, bias=False), # out: 1024 x 8 x 8
            nn.BatchNorm2d(nef*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Introducing a Flatten Layer
        self.flatten = nn.Flatten(start_dim=1) # Flattens to [50, 1024x8x8]

        # Unflatten Layer for input
        self.unflatten = View(dim=1, shape=(1, 128, 128))

        # Linear Section
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 8 * nef*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, nz) # out: nz length latent vector
        )

    def forward(self, X):
        # Go through the different sections
        X = self.unflatten(X)
        X = self.encoder_conv(X)
        X = self.flatten(X)
        X = self.encoder_lin(X)
        return X

# Creating the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_lin = nn.Sequential(
            nn.Linear(nz + nf + img_size*img_size, ndf*128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(ndf*128, ndf//16 * 32 * 32),
        )

        self.unflatten = View(dim=1, shape=(ndf//16, 32, 32)) # out: 8 x 32 x 32

        # Flatten layer to be used at end to go from [50, 1, 1, 1] to [50, 1]
        self.flatten = nn.Flatten(start_dim=1)

        self.discriminator_conv = nn.Sequential(
            nn.Conv2d(ndf//16, ndf//8, kernel_size=4, stride=2, padding=1, bias=False), # out: 16 x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf//8, ndf//4, kernel_size=4, stride=2, padding=1, bias=False), # out: 32 x 8 x 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf//4),

            nn.Conv2d(ndf//4, ndf//2, kernel_size=4, stride=2, padding=1, bias=False), # out: 64 x 4 x 4
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ndf//2),

            nn.Conv2d(ndf//2, 1, kernel_size=4, stride=1, padding=0, bias=False), # out: 128 x 2 x 2
            nn.Sigmoid()
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(ndf),
# 
            # nn.Conv2d(ndf, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, X, z, c):
        Xzc = torch.cat([X, z, c], dim=1)

        Xzc = self.discriminator_lin(Xzc)
        Xzc = self.unflatten(Xzc)
        Xzc = self.discriminator_conv(Xzc)
        Xzc = self.flatten(Xzc)
        return X

# Loss functions
def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)
    return -torch.mean(loss)

def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

# Weights
def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

# Initiating the models to GPU
G = Generator().to(device)
E = Encoder().to(device)
D = Discriminator().to(device)

# Apply the weights
E.apply(init_weights)
G.apply(init_weights)
D.apply(init_weights)

#optimizers with weight decay
optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), 
                                lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
optimizer_D = torch.optim.Adam(D.parameters(), 
                               lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

# Load in the training and validation datasets
train_data, test_data = load_dataset()

# Initiating lists to plot the loss function
lossD_list = []
lossEG_list = []
epoch_list = []

# Speeding up for conv models
# Around a 10% speedup when using this
torch.backends.cudnn.benchmark = True

# Begin training over n_epochs
for epoch in range(n_epochs):

    # Initiate the loss variables
    D_loss_acc = 0.
    EG_loss_acc = 0.

    D.train()
    E.train()
    G.train()

    for i, (images, labels) in enumerate(tqdm(train_data)):
        images = images.to(device)
        images = F.normalize(images, dim=2)
        images = images.reshape(images.size(0),-1)

        # One-hot encoding the labels
        c = torch.zeros(images.size(0), nf, dtype=torch.float32).to(device)
        c[torch.arange(images.size(0)), labels] = 1

        # Creates an nz-sized noise latent vector
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
    lossEG_list.append(EG_loss_acc / i)
    epoch_list.append(epoch+1)

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Avg_Loss_D: {:.4f}, Avg_Loss_EG: {:.4f}'
              .format(epoch + 1, n_epochs, D_loss_acc / i, EG_loss_acc / i))
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

            fig, ax = plt.subplots(3, n_show, figsize=(15,5))
            fig.subplots_adjust(wspace=0.05, hspace=0)
            plt.rcParams.update({'font.size': 20})
            fig.suptitle('Epoch {}'.format(epoch+1))
            fig.text(0.04, 0.75, 'G(z, c)', ha='left')
            fig.text(0.04, 0.5, 'x', ha='left')
            fig.text(0.04, 0.25, 'G(E(x), c)', ha='left')

            for i in range(n_show):
                ax[0, i].imshow(gener[i], cmap='gray')
                ax[0, i].axis('off')
                ax[1, i].imshow(real[i], cmap='gray')
                ax[1, i].axis('off')
                ax[2, i].imshow(recon[i], cmap='gray')
                ax[2, i].axis('off')
            plt.savefig('figs/conv/epoch_'+str(epoch+1)+'.jpg')
            plt.clf()
            
    
    if (epoch + 1) % 10 == 0:

        plt.rcParams.update({"figure.figsize": [10, 5]})
        plt.plot(epoch_list, lossD_list, label="Discriminator")
        plt.plot(epoch_list, lossEG_list, label="Generator & Encoder")
        plt.title("Average loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('figs/conv/epoch_'+str(epoch+1)+'_loss.jpg')
        plt.clf()








# train_loader, test_loader = load_dataset(batch_size=50)
# images, labels = next(iter(train_loader))

# images=images
# images2D = images.reshape(images.size(0),-1)

# c = torch.zeros(images.size(0), 6, dtype=torch.float32)
# c[torch.arange(images.size(0)), labels] = 1

# z = torch.rand(images.size(0), 100)
# z = z

# zc = torch.cat([z,c], dim=1)

# import ipdb
# ipdb.set_trace()

# gen_img = G(zc)

# gen_img = gen_img.reshape(images.size(0),-1)

# images = images.permute(0,3,1,2)
# grid = torchvision.utils.make_grid(images)

# Ez = E(images)


# tb.add_image("images", grid)
# tb.add_graph(D, torch.cat([gen_img, Ez, c], dim=1))
#tb.add_graph(D, images)
# tb.close()