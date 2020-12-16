import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms

sequence_length = 128
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epoch = 2
learning_rate = 0.003

transform = transforms.Compose([
    transforms.ToTensor(),
    #   transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

train_dataset = datasets.MNIST(root='../resources/', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = train_dataset = datasets.MNIST(root='../resources/', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)
rnn.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        image = images[0]
        images = Variable(images.view(-1, input_size, input_size)).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 109 == 0:
            print("Epoch [%d/%d],Step[%d/%d],d_loss:%.4f" % \
                  (epoch, 2, i + 1, len(train_dataset) // batch_size, loss))

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, input_size, input_size)).cuda()
    labels = Variable(labels).cuda()
    outputs = rnn.forward(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()

print("test accuracy of the model on the 10000 test images :%d%%" % (100 * correct / total))
