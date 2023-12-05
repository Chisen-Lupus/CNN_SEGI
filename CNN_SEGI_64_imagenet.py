import os
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

import torch.multiprocessing as mp

import torchvision.datasets as datasets

from utils import *

import time


# define device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# #数据预处理方法
# im_tfs = tfs.Compose([
#     tfs.ToTensor(),
#     tfs.Normalize([0.5], [0.5]) # 标准化
# ])

# train_set = MNIST('./mnist', transform=im_tfs)
# train_data = DataLoader(train_set, batch_size=128, shuffle=True)

batch = 128

# print("generate random sample")
# train_data = [[torch.rand(batch,1,64,64),torch.ones(batch,1,1,1)]]*128
# print("done")

input_file = "imagenet/train_data_batch_1"
train_set = unpickle(input_file)
train_data = ((torch.from_numpy(train_set['data'])[:128000].reshape([128000//batch, batch, 3, 64, 64])[:,:,0,:,:].unsqueeze(2))/128-1).to(device) # batch=128, total 128116
# print(train_data)

# define GI



# 定义网络
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(# 64 64 1

            nn.Upsample(scale_factor=2),

            nn.Conv2d(1, 16, 6, 2, 1),# 64 64 4 input layer, output layer, kernel, stride, padding
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3, 2, 0),# 64 64 4 input layer, output layer, kernel, stride, padding
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 64, 3, 2, 0),# 64 64 4 input layer, output layer, kernel, stride, padding
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 256, 3, 2, 0),# 64 64 4 input layer, output layer, kernel, stride, padding
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 1024, 3, 2, 0),# 64 64 4 input layer, output layer, kernel, stride, padding
            nn.ReLU(True),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 4096, 3, 2, 0),# 64 64 4 input layer, output layer, kernel, stride, padding
            nn.ReLU(True),
            nn.BatchNorm2d(4096),

            # nn.Conv2d(4096, 16384, 3, 2, 0),# 64 64 4 input layer, output layer, kernel, stride, padding
            # nn.ReLU(True),
        )
        
        self.decoder = nn.Sequential(
            
            # nn.ConvTranspose2d(16384, 4096, 3, 2, 0), # 2 2 1024
            # nn.ReLU(True),
            
            nn.ConvTranspose2d(4096, 1024, 3, 2, 0), # 2 2 1024
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            
            nn.ConvTranspose2d(1024, 256, 3, 2, 0), # 2 2 1024
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 64, 3, 2, 0), # 2 2 1024
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 16, 3, 2, 0), # 2 2 1024
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            
            nn.ConvTranspose2d(16, 16, 3, 2, 0), # 2 2 1024
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            
            nn.ConvTranspose2d(16, 2, 6, 2, 1), # 2 2 1024

            nn.MaxPool2d(2),

            # nn.Tanh()
        )

    def forward(self, x):

        encode = self.encoder(x)

        # print(encode.shape)
        decode = self.decoder(encode)

        # print(decode.shape)
        return encode, decode
        
net = autoencoder()
# net.load_state_dict(torch.load("./E7.pth"))
net.to(device)

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#                                change lr here

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x
    
# 开始训练自动编码器
time_start=time.time()
iter = 0
for epoch in range(10):
    for im in train_data:
        # im = im.view(im.shape[0], -1)# used in linear net
        # print(im.shape)
        # im = im.to(device)
        im = Variable(im)
        # print(im.shape)
        # 前向传播
        _, output = net(im)
        # output = output.view(im.shape[0], 1, 64, 64)# used in linear net
        # print(output,"and", im)
        loss = criterion(output[:,0,:,:].unsqueeze(1), im) / im.shape[0] # 平均
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter%100==0: 
            print("epoch:", epoch, "iter:", iter, ", loss:", loss)
        iter+=1
    
    # if (e+1) % 20 == 0: # 每 20 次，将生成的图片保存一下
    #     print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data))
    #     pic = to_img(output.cpu().data)
    #     if not os.path.exists('./simple_autoencoder'):
    #         os.mkdir('./simple_autoencoder')
    #     save_image(pic, './simple_autoencoder/image_{}.png'.format(e + 1))
time_end=time.time()
print('time cost',time_end-time_start,'s')


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#matplotlib inline


print("Plot data")
# view_data = torch.from_numpy(plt.imread("C:/Users/Chisen/OneDrive - University of Macau/Coursework/Ghost imaging/experiments/sample_image/baboon-64-gray.png")[:,:,0]).reshape(1,1,64,64).to(device)
view_data = train_data[0].to(device)
# view_data = torch.rand(64,64).reshape(1,1,64,64).to(device)
encode, _ = net(view_data)
plt.subplot(1,6,1)
plt.title("image")
plt.imshow(view_data[0,0,:,:].cpu().detach().numpy(),plt.cm.gray)
plt.subplot(1,6,2)
plt.title("pattern")
plt.imshow(_[0,0,:,:].cpu().detach().numpy(),plt.cm.gray)
plt.subplot(1,6,3)
plt.title("pattern2")
plt.imshow(_[0,1,:,:].cpu().detach().numpy(),plt.cm.gray)
# plt.imshow(encode[:,:,0].cpu().detach().numpy(),plt.cm.gray)
plt.subplot(1,6,4)
plt.subplot(1,6,5)
plt.show()






# torch.save(net.state_dict(), './E8.pth')


# 可视化结果
# view_data = Variable((train_set.train_data[:200].type(torch.FloatTensor).view(-1, 28*28) / 255. - 0.5) / 0.5) # used in linear
# for view_data, _ in train_data:
#     view_data = Variable(view_data)
#     view_data, _ = view_data.to(device), _.to(device)
#     break
# encode, _ = net(view_data)    # 提取压缩的特征值
# encode, _ = encode.cpu(), _.cpu()
# encode = encode.squeeze()
# print(encode.shape)
# fig = plt.figure(2)
# ax = Axes3D(fig)    # 3D 图
# # x, y, z 的数据值
# X = encode.data[:, 0].numpy()
# Y = encode.data[:, 1].numpy()
# Z = encode.data[:, 2].numpy()
# values = 0#train_set.train_labels[:200].numpy()  # 标签值
# for x, y, z, s in zip(X, Y, Z, values):
#     c = cm.rainbow(int(255*s/9))    # 上色
#     ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())
# ax.set_zlim(Z.min(), Z.max())
# plt.show()

# input = [1.19, -3.36, 2.06]

# def testnet(input):
#     code = Variable(torch.FloatTensor(input)).reshape(1,3,1,1)   # 给一个 code 是 (1.19, -3.36, 2.06)
#     decode = net.decoder(code.to(device))
#     decode_img = to_img(decode).squeeze()
#     decode_img = decode_img.data.cpu().numpy() * 255
#     plt.imshow(decode_img.astype('uint8'), cmap='gray') # 生成图片 3
#     plt.show()

# testnet(input)