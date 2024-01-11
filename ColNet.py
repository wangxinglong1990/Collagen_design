import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)

class Attn(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= (1,1) )
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= (1,1) )
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= (1,1) )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

class ResBlock(nn.Module):
    def __init__(self,chanels):
        super(ResBlock, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(chanels, chanels, kernel_size =(3,3), padding=(1,1)),nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(chanels, chanels, kernel_size =(3,3), padding=(1,1)),nn.ReLU())
    def forward(self,x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        out = x + x2
        return out

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size =(3,3), stride = (3,3),padding=0),nn.BatchNorm2d(64), nn.ReLU(),nn.Conv2d(64, 128, 1, padding=0),nn.BatchNorm2d(128),nn.ReLU())
        self.res1 = ResBlock(128)
        self.conv1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = (3,3), stride = 1, padding=0),nn.BatchNorm2d(256), nn.ReLU())       
        #self.res2 = ResBlock(256)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 512, kernel_size = (3,3), stride = 1, padding=0),nn.BatchNorm2d(512), nn.ReLU())
        #self.res3 = ResBlock(512)
        self.conv3 = nn.Sequential((nn.Conv2d(512, 1024, kernel_size = (3,3), stride = 1, padding=0)), nn.ReLU())
        self.attn = Attn(1024)
        self.dense1 = nn.Linear(15360,1024)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(1024,1)

    def forward(self,x, bz):
        x1 = self.double_conv(x)
        x2 = self.res1(x1)
        x2 = self.conv1(x2)  
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x4 = self.attn(x4)
        x4 = x4.reshape(bz, 15360)
        x5 = self.dense1(x4)
        x6 = self.relu(x5)
        out = self.dense2(x6)
        return out
