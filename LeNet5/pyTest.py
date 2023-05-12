import torch
from torch import nn

if __name__ == '__main__':

    input = torch.randn(32, 1, 5, 5) # # （batch_size,channels,height,width）
    # With default parameters
    m = nn.Flatten()
    output = m(input)
    output.size()
    # torch.Size([32, 25])
    # With non-default parameters
    m = nn.Flatten(0, 2)
    output = m(input)
    output.size()
    # torch.Size([160, 5])