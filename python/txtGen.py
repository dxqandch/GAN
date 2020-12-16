import os

dir = '../resources/data/'

files = os.listdir(dir)  # 返回目录下的所有文件和文件夹的名字
train = open('../temp/train.txt', 'a')
val = open('../temp/val.txt', 'a')
i = 1  # 区分训练数据和测试数据
for file in files:
    if not file.endswith("png"):
        continue
    if file.startswith("fake"):
        label = 0
    else:
        label = 1
    if i % 10 != 0:
        fileType = os.path.split(file)
        print(fileType)
        if fileType[1] == '.txt':
            continue
        name = str(dir) + file + ' ' + str(int(label)) + '\n'
        train.write(name)
        i = i + 1
        print(i)
    else:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = str(dir) + file + ' ' + str(int(label)) + '\n'
        val.write(name)
        i = i + 1
        print(i)
