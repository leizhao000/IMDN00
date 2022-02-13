from model import architecture
import torch.nn as nn
#model = mdasr1.MDASR(in_channels=3,out_channels=64)
model = architecture.IMDN()
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

print_network(model)