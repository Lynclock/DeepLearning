import torch
from torch import nn
from LeNet import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets,transforms
import os

# 数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True) #shuffle 顺序是否打乱

#加载测试数据集
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用LeNet里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)

# 定义一个损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer =torch.optim.SGD(model.parameters(),lr=1e-3, momentum=0.9) #momentum 用于梯度下降的动量参数，防止收敛在局部最小

# 学习率每隔10轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

# 定义训练函数
def train(dataloader,model,loss_fn,optimizer):
        loss, current, n = 0.0,0.0,0
        for batch, (X,y) in enumerate(dataloader):
            # 前向传播
            X,y = X.to(device),y.to(device)
            output = model(X)
            cur_loss = loss_fn(output,y) #真实值与标签进行交叉熵计算得到损失值
            _, pred = torch.max(output,axis=1) #输出最大概率对应的索引

            cur_acc = torch.sum(y==pred)/output.shape[0]

            optimizer.zero_grad() # 梯度清零，每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零。
            cur_loss.backward() # 反向传播，指的是计算神经网络参数梯度的方法。该方法根据微积分中的链式规则，按相反的顺序从输出层到输入层遍历网络。 该算法存储了计算某些参数梯度时所需的任何中间变量（偏导数）
            optimizer.step() #梯度更新，在训练深度神经网络的过程中，我们需要通过反向传播算法计算每一个参数对损失函数的梯度，然后使用优化器更新参数，使得损失函数最小化。而optimizer.step()方法就是用于执行参数更新的。

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n+1 #批次
        print("train_loss" + str(loss/n))
        print("train_acc" + str(current/n))

def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad(): # 表明当前计算不需要反向传播
        for batch,(X,y) in enumerate(dataloader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)  # 真实值与标签进行交叉熵计算得到损失值
            _, pred = torch.max(output, axis=1)  # 输出最大概率对应的索引

            cur_acc = torch.sum(y==pred)/output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n+1 #批次
        print("val_loss" + str(loss/n))
        print("val_acc" + str(current/n))

        return current/n
# 开始训练
epoch = 50
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n---------------')
    train(train_dataloader,model,loss_fn,optimizer)
    a = val(test_dataloader,model,loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(),'save_model/best_model.pth')
print('Done!')