import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from mnist_modules import AutoEncoder
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image


kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./', train=False, transform=transforms.ToTensor()),
    batch_size=32, shuffle=True, **kwargs)

test_data = list(test_loader)

model = AutoEncoder().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]


def train(epoch):
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        x = Variable(data, requires_grad=False).cuda()
        x_tilde, z_e_x, z_q_x = model(x)

        z_q_x.retain_grad()

        loss_recons = F.l1_loss(x_tilde, x)
        loss_e1 = torch.pow(z_e_x.detach() - z_q_x, 2).sum(1).mean()
        loss_e2 = torch.pow(z_e_x - z_q_x.detach(), 2).sum(1).mean()

        opt.zero_grad()
        loss_recons.backward(retain_graph=True)
        z_e_x.backward(z_q_x.grad.data, retain_graph=True)
        (loss_e1 + 0.25*loss_e2).backward()

        # for param in model.parameters():
        #     param.grad.data.clamp_(-1., 1.)

        opt.step()

        train_loss.append(to_scalar([loss_recons, loss_e1]))

        print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            np.asarray(train_loss).mean(0))


def test():
    x = Variable(test_data[0][0]).cuda()
    x_tilde, _, _ = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = x_cat.cpu().data
    save_image(images, './test.png', nrow=8)


for i in xrange(50):
    train(i)
    test()
