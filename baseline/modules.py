import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy


def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]


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
            nn.BatchNorm2d(64),
        )

        self.embedding = nn.Embedding(512, 64)
        self.embedding.weight.data.copy_(1./512 * torch.randn(512, 64))

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z_e_x = self.encoder(x)
        B, C, H, W = z_e_x.size()

        z_e_x_transp = z_e_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        emb = self.embedding.weight.transpose(0, 1)  # (C, K)
        dists = torch.pow(
            z_e_x_transp.unsqueeze(4) - emb[None, None, None],
            2
        ).sum(-2)
        latents = dists.min(-1)[1]

        z_q_x = self.embedding(latents.view(latents.size(0), -1))
        z_q_x = z_q_x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_e_x, z_q_x


if __name__ == '__main__':
    model = AutoEncoder().cuda()
    model.zero_grad()
    x = Variable(torch.randn(32, 1, 28, 28).cuda(), requires_grad=False)
    x_tilde, z_e_x, z_q_x = model(x)
    z_q_x.retain_grad()

    loss_1 = F.binary_cross_entropy(x_tilde, x)
    loss_1.backward(retain_graph=True)
    assert model.encoder[-2].bias.grad is None
    model.embedding.zero_grad()
    z_e_x.backward(z_q_x.grad, retain_graph=True)
    assert model.embedding.weight.grad.sum().data.cpu().numpy()[0] == 0
    bias = deepcopy(model.encoder[-2].bias.grad.data)

    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_2.backward(retain_graph=True)
    emb = deepcopy(model.embedding.weight.grad.data)
    assert (bias == model.encoder[-2].bias.grad.data).all() is True

    loss_3 = 0.25*F.mse_loss(z_e_x, z_q_x.detach())
    loss_3.backward()
    assert (emb == model.embedding.weight.grad.data).all() is True

    print loss_1, loss_2, loss_3
