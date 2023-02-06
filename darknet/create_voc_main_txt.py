
import os
import random

if __name__ == '__main__':
    trainval_percent = 0.9
    train_percent = 1
    path = r'E:\MyData\tello_voc_data\VOCdevkit2007\VOC2007'
    xmlfilepath = path + r'\Annotations'
    txtsavepath = path + r'\ImageSets\Main'
    # 读取xmlfilepath路径下所有的xml文件的名字
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(txtsavepath + r'\trainval.txt', 'w')
    ftest = open(txtsavepath + r'\test.txt', 'w')
    ftrain = open(txtsavepath + r'\train.txt', 'w')
    fval = open(txtsavepath + r'\val.txt', 'w')
    # 将所有数据分为trainval和test，trainval_percent表示trainval的百分比，剩下的是test,train_percent是trainval中train的百分比，剩下的是val
    # 将train_percent设为1，那么val中则没有数据
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
