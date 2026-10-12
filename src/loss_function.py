import torch
from torch import nn

torch.set_default_dtype(torch.float32)

class Cos_loss(nn.Module):
    def __init__(self,margin):
        super(Cos_loss, self).__init__()
        self.margin = margin
        self.cossim = nn.CosineSimilarity(dim=2)

    def forward(self,z1,z2,l):

        similarity = self.cossim(z2,z1).requires_grad_(True)
        #when using with att
        similarity = similarity.mean(dim=1)
        test = (l==1).float()
        Cosloss = torch.zeros(similarity.size())
        cosembloss=0.
        for i in range(len(test)):
            if test[i]==1:
                Cosloss[i] = similarity[i]
            else:
                Cosloss[i] = torch.clamp(similarity[i] - self.margin, min=0.0)

        Cosloss_tensor = Cosloss.clone().requires_grad_(True)

        loss = torch.mean(Cosloss_tensor)

        return loss

class Dis_loss(nn.Module):
    def __init__(self,margin):
        super(Dis_loss, self).__init__()
        self.margin = margin

    def forward(self,dis,l):


        Cosloss_tensor = torch.square(dis-l)*0.5

        loss = torch.mean(Cosloss_tensor)

        return loss
