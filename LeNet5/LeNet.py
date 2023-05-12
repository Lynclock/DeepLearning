import torch
from torch import nn

# 定义一个网络模型
class MyLeNet5(nn.Module):
    #初始化网络
    def __init__(self):
        super(MyLeNet5,self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5,padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c5 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)

        self.flatten = nn.Flatten() # 将连续的维度范围展平为张量
        # 我们用一个r×r×16*120的filter去卷积激活函数的输出，得到的结果就是一个全连接层的一个神经元的输出，这个输出就是一个值
        # 我们用一个1×1×120*84的filter去卷积激活函数的输出
        # 我们用一个1×1×84*10的filter去卷积激活函数的输出
        self.f6 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)

    def forward(self,x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)


        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

if __name__ == '__main__':

    x = torch.rand(1,1,28,28) # （batch_size,channels,height,width）
    model =MyLeNet5()
    y = model(x)


