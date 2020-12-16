root = '../resources/'
source_file = 'train.txt'
target_file = 'train1.txt'
f = open(root + source_file, "r")
f1 = open(root + target_file, "w")
for line in f:
    line=line.strip("\n")
    line = line + " 0\n"
    f1.write(line)
# for line in f:
#     line = line + ' 0'
#     print(line)
