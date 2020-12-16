import math

import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms

unloader = transforms.ToPILImage()

MyTransforms = transforms.Compose(
    [transforms.ToTensor(),  # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated


# 不用MNIST数据集，使用自己的数据集
# train_data = torchvision.datasets.MNIST(root="../resources",
#                                         train=True,
#                                         transform=torchvision.transforms.ToTensor(),
#                                         download=DOWLOAD_MNIST
#                                         )
# print(train_data.data.size())

# test_data = torchvision.datasets.MNIST(root="../resources",
#                                        train=False,
#                                        transform=torchvision.transforms.ToTensor(),
#                                        download=DOWLOAD_MNIST
#                                       )
# test_x = Variable(torch.unsqueeze(test_data, dim=1), volatile=True).type(torch.FloatTensor)
# test_y = test_data.test_labels
# print(test_x)
# print(test_data.test_labels)


EPOCH = 30
BATCH_SIZE = 20
LR = 0.001
DOWLOAD_MNIST = True

root = '../temp/'  # 数据集的地址 txt文件的地址 txt文件保存了(图片路径，标签)


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        """
可对比torchvision中的visionDataset的实现查看
        :param txt: 文件路径
        :param transform: 转换为Tensor，一般该参数值为ToTensor
        :param target_transform:
        :param loader:
        """
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # todo [图片路径,标签]，下文words【0】指的图片路径或图片文件？  done:就是路径，并在getitem中利用该路径得到图片
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))
            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
        # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)
        # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)
            # 数据标签转换为Tensor
        return img, label
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


train_data = MyDataset(txt=root + 'train.txt', transform=MyTransforms)
print(train_data.imgs)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = MyDataset(txt=root + "val.txt", transform=MyTransforms)

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


# input[20,3,324,392]
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.fn = nn.Linear(47040, 50)
        self.out = nn.Linear(51200, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        #   fn = self.fn(x)
        output = self.out(x)
        return output


cnn = CNN().cuda()
print(cnn)

params = list(cnn.parameters())
print(list(cnn.named_parameters()))
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()

        output = cnn(b_x)
        loss = loss_function(output, b_y)  # todo output:(20,2),b_y:(20)为什么能算loss？
        arr1 = output.cpu().detach().numpy()
        arr2 = b_y.cpu().detach().numpy()
        res = 0.0
        for i in range(2):
            for j in range(20):
                res = -math.log(math.fabs(arr1[j][i])) * float(arr2[j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2 == 0:  # 每经过step*batchsize个训练数据就测试一次新的网络
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.cuda(), test_y.cuda()
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = float(sum(pred_y == test_y)) / test_y.size(0)
                print("Epoch:", epoch, "|Step:", step,
                      "|train loss:%.4f" % loss, "|test accuracy:%.4f" % accuracy)
    for test_x, test_y in test_loader:
        test_x, test_y = test_x.cuda(), test_y.cuda()
        test_output = cnn(test_x[:10])
        pred_y = []
        for x, y in test_output:
            if x > y:
                pred_y.append(0)
            else:
                pred_y.append(1)
        print(pred_y[:10], "prediction sort")
        print(test_y[:10], 'real sort')

        print({x: y for x, y in zip(pred_y, test_y) if x > y}, "以假乱真")
        a = {index: (a, b) for index, (a, b) in enumerate(zip(pred_y, test_y)) if a > b}
        for index, (x, y) in a.items():
            imshow(test_x[index])

        print({x: y for x, y in zip(pred_y, test_y) if x < y}, "误伤友军")
