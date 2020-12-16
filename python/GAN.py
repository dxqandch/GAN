import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


# 导入模块
def to_var(x):
    x = x.cuda()
    return Variable(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

mnist = datasets.MNIST(root='../resources/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=100, shuffle=True)

D = nn.Sequential(
    nn.Linear(784 * 3, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

G = nn.Sequential(
    nn.Linear(64, 128),
    nn.LeakyReLU(0.2),
    nn.Linear(128, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784 * 3),
    nn.Tanh()
)

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(200):
    for i, (images, _) in enumerate(data_loader):  # enumerate:返回下标和数据
        batch_size = images.size(0)
        images = to_var(images.view(batch_size, -1))
        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var((torch.zeros(batch_size)))

        outputs = D(images)
        outputs1 = outputs.view(-1)
        d_loss_real = criterion(outputs1, real_labels)
        real_score = outputs

        z = to_var(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)
        outputs2 = outputs.view(-1)
        d_loss_fake = criterion(outputs2, fake_labels)
        fake_score = outputs

        d_loss = d_loss_fake + d_loss_real
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        z = to_var(torch.randn(batch_size, 64))  # todo once again?
        fake_images = G(z)
        outputs = D(fake_images)
        outputs = outputs.view(-1)
        g_loss = criterion(outputs, real_labels)
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 300 == 0:
            print("Epoch [%d/%d],Step[%d/%d],d_loss:%.4f,g_loss:%.4f,D(x):%.2f,D(G(z):%.2f" % \
                  (epoch, 200, i + 1, 600, d_loss.item(), g_loss.item(),
                   real_score.data.mean(), fake_score.data.mean()))
            images = images.view(images.size(0), 3, 28, 28)
            save_image(denorm(images.data), "../resources/data/real_images-%d-%d.png" % (epoch + 1, (i + 1)/ 300))
            fake_images = fake_images.view(fake_images.size(0), 3, 28, 28)
            save_image(denorm(fake_images.data), "../resources/data/fake_images-%d-%d.png" % (epoch + 1, (i + 1)/300))

torch.save(G.state_dict(), '../resources/generator.pkl')
torch.save(D.state_dict(), "../resources/discriminator.pkl")
