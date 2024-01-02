import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ColDiff import UNet
import os
import sequence_encode
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))



def train(epochs,bz,lr):

    ##create folder and loss text
    f = open('epoch_loss.txt','w')
    f.close()
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    ##load data
    f = open(r'hcollagen_clean1.txt')
    content = f.readlines()
    f.close()
    seqs = []
    for i in content:
        seqs.append(i.strip('\n'))
    seq_data = sequence_encode.get_input(seqs = seqs,seq_lenth=30)
    real_bz = bz
    loader = DataLoader(dataset=seq_data,batch_size=real_bz,shuffle=True,drop_last= True)

    ##model initiation
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = UNet(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(device=device)

    for epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for i, images in enumerate(loader):
            count += 1
            images = images.to(device)
            images = images.reshape(real_bz,1,21,30)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            epoch_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f = open('epoch_loss.txt','a+')
        loss_ave = 'epoch_loss %s\n'%(epoch_loss/count)
        f.write(loss_ave)
        f.close()
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), r"saved_models/" + 'epoch_%s.pt'%(str(epoch)))
        print(epoch)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=2000)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-bz', type=int, default=32)
    
    args = parser.parse_args()
    epoch = args.epoch
    lr = args.lr
    bz = args.bz

    train(epoch, lr, bz)
