import torch
from torch import nn
import torch.nn.functional as F
#from torchinfo import summary
#from pthflops import count_ops

torch.set_default_dtype(torch.float32)

class DiscNN(nn.Module):
    def __init__(self, input_size,drop_rate,discout):
        super(DiscNN, self).__init__()
        self.hid1=nn.Linear(1, discout)
        self.hid2=nn.Linear(discout, discout)

        self.drop = nn.Dropout(p=drop_rate)
        self.discout = discout

    def forward(self, x):

        hidden1= F.gelu(self.hid1(x))
        hidden2= F.gelu(self.hid2(hidden1))

        return hidden2


class ClassifierNN(nn.Module):
    def __init__(self, input_size,f,drop_rate):
        super(ClassifierNN, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        """
        for more than 3 features
        self.hid1 = nn.Linear(f*input_size, f*input_size)
        self.hid2 = nn.Linear(f*input_size, f*input_size//2)
        self.out = nn.Linear(f*input_size//2, 1)
        """
        """
        for land_mine
        self.hid1 = nn.Linear(f*input_size, f*input_size//2)
        self.hid2 = nn.Linear(f*input_size//2, f*input_size//4)
        self.out = nn.Linear(f*input_size//4, 1)
        self.gelu=nn.GELU()
        """
        self.hid1 = nn.Linear(f*input_size, f*input_size//2)
        self.hid2 = nn.Linear(f*input_size//2, f*input_size//4)
        self.out = nn.Linear(f*input_size//4, 1)
        self.gelu=nn.GELU()

    def forward(self, z):
        z = z.flatten(start_dim=1)
        chidden1= self.gelu(self.hid1(z))
        chidden1 = self.drop(chidden1)
        chidden2= self.gelu(self.hid2(chidden1))
        chidden2 = self.drop(chidden2)
        cout= self.out(chidden2)

        return cout

class DistanceNN(nn.Module):
    def __init__(self, input_size):
        super(DistanceNN, self).__init__()
        self.hid1 = nn.Linear(input_size*2, 128)
        self.hid2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, z1, z2):
        x=torch.concatenate((z1,z2),2)
        dhidden1= F.gelu(self.hid1(x))
        dhidden2= F.gelu(self.hid2(dhidden1))
        dout= F.tanh(self.out(dhidden2))
        return dout

class AttentionNN(nn.Module):
    def __init__(self, input_size):
        super(AttentionNN, self).__init__()
        self.query = nn.Linear(input_size,input_size)
        self.key = nn.Linear(input_size,input_size)
        self.value = nn.Linear(input_size,input_size)
        self.softmax = nn.Softmax(dim=-1)
        self.input_size = input_size


    def forward(self, z1):
        q = self.query(z1)
        k = self.key(z1)
        scores = torch.matmul(q, k.transpose(1,2))
        scores = scores/ torch.sqrt(torch.tensor(q.size(-1)).float())
        weights = self.softmax(scores)

        v = self.value(z1)

        attended1 = torch.matmul(weights,v)


        return attended1,weights
