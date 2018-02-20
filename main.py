import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from mnist_modules import AutoEncoder, to_scalar
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time


kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './data/fashion_mnist/', train=True, download=True,
        transform=transforms.ToTensor()
        ), batch_size=64, shuffle=False, **kwargs
    )

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './data/fashion_mnist/', train=False,
        transform=transforms.ToTensor()
    ), batch_size=32, shuffle=False, **kwargs
)
test_data = list(test_loader)

model = AutoEncoder().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).cuda()

        opt.zero_grad()

        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad()

        loss_1 = F.binary_cross_entropy(x_tilde, x)
        # loss_1 = F.l1_loss(x_tilde, x)
        loss_1.backward(retain_graph=True)
        model.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_2.backward(retain_graph=True)

        loss_3 = 0.25 * F.mse_loss(z_e_x, z_q_x.detach())
        loss_3.backward()
        opt.step()

        train_loss.append(to_scalar([loss_1, loss_2]))

        print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            np.asarray(train_loss).mean(0),
            time.time() - start_time    
        )


def test():
    x = Variable(test_data[0][0]).cuda()
    x_tilde, _, _ = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = x_cat.cpu().data
    save_image(images, './test.png', nrow=8)


for i in xrange(100):
    train(i)
    test()
