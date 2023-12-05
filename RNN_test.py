# https://cloud.tencent.com/developer/article/1397609

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2019)

# 超参设置
TIME_STEP = 10  # RNN时间步长
INPUT_SIZE = 1  # RNN输入尺寸
INIT_LR = 0.01  # 初始学习率
N_EPOCHS = 100  # 训练回数


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,  # RNN隐藏神经元个数
            num_layers=1,  # RNN隐藏层个数
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        out, h = self.rnn(x, h)
        prediction = self.out(out)
        return prediction, h


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=INIT_LR)
loss_func = nn.MSELoss()
h_state = None  # 初始化隐藏层

yy = np.zeros([0])
pp = np.zeros([0])

plt.figure()
plt.ion()
for step in range(N_EPOCHS):
    start, end = step * np.pi, (step + 1) * np.pi  # 时间跨度
    # 使用Sin函数预测Cos函数
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])  # 尺寸大小为(time_step, batch, input_size)
    y = torch.from_numpy(y_np[:, np.newaxis, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # RNN输出（预测结果，隐藏状态）
    h_state = h_state.detach()  # 这一行很重要，将每一次输出的中间状态传递下去(不带梯度)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 绘制中间结果
    plt.cla()
    plt.plot(steps, y_np, 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.1)

    yy = np.append(yy, y_np)
    pp = np.append(pp, prediction.data.numpy().flatten())

plt.ioff()

plt.figure()
ss = np.linspace(1,1000,1000)
plt.plot(ss, yy, 'r-')
plt.plot(ss, pp, 'b-')

plt.show()

