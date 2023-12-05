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

# define device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#数据预处理方法
im_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5]) # 标准化
])

train_set = MNIST('./mnist', transform=im_tfs)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)


# 定义网络
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),#, stride=1, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2),#, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 16, 4),#, stride=2, padding=1),  # (b, 3, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2),#, stride=1),  # (b, 3, 2, 2)
            nn.Conv2d(16, 16, 3),#, stride=2, padding=1),  # (b, 3, 3, 3)
            nn.ReLU(True),
            nn.Conv2d(16, 3, 3),#, stride=2, padding=1),  # (b, 3, 3, 3)
            nn.ReLU(True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 3),
            nn.Upsample(scale_factor=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 4),
            nn.Upsample(scale_factor=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3),

            # nn.ConvTranspose2d(3, 16, 3),#, stride=2),  # (b, 16, 5, 5)
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 16, 3),#, stride=2, padding=1),  # (b, 16, 5, 5)
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 16, 3),#, stride=2, padding=1),  # (b, 16, 5, 5)
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 8, 2),#, stride=2, padding=1),  # (b, 8, 15, 15)
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 1, 2),#, stride=2, padding=2),  # (b, 1, 28, 28)
            nn.Tanh()
        )

        # #编码器
        # self.encoder = nn.Sequential(
        #     nn.Linear(28*28, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, 3) # 输出的 code 是 3 维，便于可视化
        # )
        # #解码器
        # self.decoder = nn.Sequential(
        #     nn.Linear(3, 12),
        #     nn.ReLU(True),
        #     nn.Linear(12, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 28*28),
        #     nn.Tanh()  #使输出范围在-1~1
        # )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        # print(encode.shape, decode.shape)
        return encode, decode
        
net = autoencoder()
net.to(device)

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x
    
# 开始训练自动编码器
for e in range(40):
    for im, _ in train_data:
        # im = im.view(im.shape[0], -1)# used in linear net
        im, _ = im.to(device), _.to(device)
        im = Variable(im)
        # print(im.shape)
        # 前向传播
        _, output = net(im)
        loss = criterion(output, im) / im.shape[0] # 平均
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("epoch:", e, ", loss:", loss)
    
    if (e+1) % 20 == 0: # 每 20 次，将生成的图片保存一下
        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data))
        pic = to_img(output.cpu().data)
        if not os.path.exists('./simple_autoencoder'):
            os.mkdir('./simple_autoencoder')
        save_image(pic, './simple_autoencoder/image_{}.png'.format(e + 1))

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#matplotlib inline

# 可视化结果
# view_data = Variable((train_set.train_data[:200].type(torch.FloatTensor).view(-1, 28*28) / 255. - 0.5) / 0.5) # used in linear
for view_data, _ in train_data:
    view_data = Variable(view_data)
    view_data, _ = view_data.to(device), _.to(device)
    break
encode, _ = net(view_data)    # 提取压缩的特征值
encode, _ = encode.cpu(), _.cpu()
encode = encode.squeeze()
print(encode.shape)
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D 图
# x, y, z 的数据值
X = encode.data[:, 0].numpy()
Y = encode.data[:, 1].numpy()
Z = encode.data[:, 2].numpy()
values = train_set.train_labels[:200].numpy()  # 标签值
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # 上色
    ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()





input = [1.19, -3.36, 2.06]

def testnet(input):
    code = Variable(torch.FloatTensor(input)).reshape(1,3,1,1)   # 给一个 code 是 (1.19, -3.36, 2.06)
    decode = net.decoder(code.to(device))
    decode_img = to_img(decode).squeeze()
    decode_img = decode_img.data.cpu().numpy() * 255
    plt.imshow(decode_img.astype('uint8'), cmap='gray') # 生成图片 3
    plt.show()

testnet(input)