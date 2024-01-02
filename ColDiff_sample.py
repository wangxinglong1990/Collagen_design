import numpy as np
import torch
import torch.nn as nn
from modules import UNet
import os
from sklearn.linear_model import LinearRegression
import argparse

fit_model = LinearRegression()

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
    def sample_promoters (self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, 21, 30)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def dataTogene(data, bz,name):
    data = data.reshape(bz,30,21)
    data = np.eye(data.shape[2])[data.argmax(2)]

    list = []
    for i in data:
        for l in i:
            l1 =  np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l2 =  np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l3 =  np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l4 =  np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l5 =  np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l6 =  np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l7 =  np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l8 =  np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l9 =  np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l10 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l11 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l12 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            l13 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
            l14 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
            l15 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
            l16 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
            l17 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
            l18 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
            l19 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
            l20 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
            l21 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])

            if str((l == l1).all()) == "True":
                list.append('H')
            if str((l == l2).all()) == "True":
                list.append('D')
            if str((l == l3).all()) == "True":
                list.append('R')
            if str((l == l4).all()) == "True":
                list.append('F')
            if str((l == l5).all()) == "True":
                list.append('A')
            if str((l == l6).all()) == "True":
                list.append('C')
            if str((l == l7).all()) == "True":
                list.append('G')
            if str((l == l8).all()) == "True":
                list.append('Q')
            if str((l == l9).all()) == "True":
                list.append('E')
            if str((l == l10).all()) == "True":
                list.append('K')
            if str((l == l11).all()) == "True":
                list.append('L')
            if str((l == l12).all()) == "True":
                list.append('M')
            if str((l == l13).all()) == "True":
                list.append('N')
            if str((l == l14).all()) == "True":
                list.append('S')
            if str((l == l15).all()) == "True":
                list.append('Y')
            if str((l == l16).all()) == "True":
                list.append('T')
            if str((l == l17).all()) == "True":
                list.append('I')
            if str((l == l18).all()) == "True":
                list.append('W')
            if str((l == l19).all()) == "True":
                list.append('P')
            if str((l == l20).all()) == "True":
                list.append('V')
            if str((l == l21).all()) == "True":
                list.append('O')
    f = open ('output_collagen%s.txt'%name,'w')

    #all_seqs = []
    for i in range (bz):
        seq =(list[(i*30):((i+1)*(30))])
        seq = str(seq).replace(',','').replace("'",'').replace(' ','').replace('[','').replace(']','')
        f.write('>%s'%i)
        f.write('\n')
        f.write(seq)
        f.write('\n')
        #all_seqs.append(seq)
    f.close()
    #all_seqs = np.array(all_seqs)
    #all_seqs = all_seqs.reshape([len(all_seqs),])
    #np.save('denovo_seq%s.npy'%num,all_seqs)

def collagenGXY(path1 = 'output_collagen.txt', path2= 'collagen_count.txt', path3='count_freq.txt', bz= 512):
    f = open(r'%s'%path1)
    a = f.readlines()
    f.close()
    c = 0
    t_count = 0
    f_count = 0
    f = open ('%s'%path2,'w')
    f.close()
    f = open ('%s'%path2,'a+')
    for i in a:
        c += 1
        if c % 2 == 0:
            if i[0] == 'G' and i[3] == 'G' and i[6] == 'G' and i[9] == 'G' and i[12] == 'G' and i[15] == 'G' and i[18] == 'G' and i[21] == 'G' and i[24] == 'G' and i[27] == 'G':
                t_count += 1
                wr1 = 'True' + ' ' + str(c-1) + ' ' + i
                f.write(wr1)
            else:
                f_count += 1
                wr2 = 'False' + ' ' + str(c-1) + ' ' + i
                f.write(wr2)
    f.close()
    t_freq = t_count/bz
    f_freq = f_count/bz
    f = open ('%s'%path3,'w')
    f.close()
    f = open ('%s'%path2,'a+')
    f.write('True %s \n'%str(t_freq))
    f.write('False %s \n'%str(f_freq))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-model', type=str, default='ColDiff_model.pt')
    parser.add_argument('-bz', type=int, default=512)
    parser.add_argument('-file_name', type=str, default='outputCMPs.txt')
    
    args = parser.parse_args()
    device = args.device
    model = args.model
    bz = args.bz
    file_name = args.file_name
    
    device = torch.device(device)
    model = UNet(device=device).to(device)
    load_model = torch.load(model)
    model.load_state_dict(load_model)
    diffusion = Diffusion(device=device)
    data = diffusion.sample_promoters(model, bz).cpu().detach().numpy()
    dataTogene(data=data,bz=bz,name=file_name)
    #collagenGXY()

