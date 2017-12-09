import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_cuda_variable(array, requires_grad=True, volatile=False):
    if 'numpy' in str(type(array)):
        array = torch.from_numpy(array)
    return Variable(array,
                    requires_grad=requires_grad,
                    volatile=volatile).cuda()


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, 1)
        )
    
    def forward(self, x):
        return x + self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.BatchNorm2d(64)
        )

        self.embedding = nn.Embedding(1024, 64)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        z_e_x = self.encoder(x)
        B, C, H, W = z_e_x.size()

        z_e_x_transp = z_e_x.transpose(1, 2).transpose(2, 3) # (B, H, W, C)
        emb = self.embedding.weight.transpose(0, 1)
        dists = torch.pow(z_e_x_transp.unsqueeze(4)-emb[None, None, None], 2).sum(-2)
        latents = dists.min(-1)[1]

        z_q_x = self.embedding(latents.view(latents.size(0), -1))
        z_q_x = z_q_x.view(B, H, W, C).transpose(2, 3).transpose(1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_e_x, z_q_x


class NPAutoEncoder(nn.Module):
    def __init__(self):
        super(NPAutoEncoder, self).__init__()
        self.encoder = nn.Embedding(60032, 7*7*64)
        self.embedding = nn.Embedding(512, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z_e_x = self.encoder(x).view(-1, 64, 7, 7)
        B, C, H, W = z_e_x.size()

        z_e_x_transp = z_e_x.transpose(1, 2).transpose(2, 3)  # (B, H, W, C)
        emb = self.embedding.weight.transpose(0, 1)
        dists = torch.pow(z_e_x_transp.unsqueeze(
            4) - emb[None, None, None], 2).sum(-2)
        latents = dists.min(-1)[1]

        z_q_x = self.embedding(latents.view(latents.size(0), -1))
        z_q_x = z_q_x.view(B, H, W, C).transpose(2, 3).transpose(1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_e_x, z_q_x


if __name__ == '__main__':
    x = torch.randn(32, 3, 28, 28).cuda()
    x = Variable(x)

    model = AutoEncoder().cuda()
    model(x)
