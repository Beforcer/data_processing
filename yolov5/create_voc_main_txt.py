
import os
import random
from tqdm import tqdm

if __name__ == '__main__':
    # 将所有数据分为trainval和test，trainval_percent表示trainval的百分比，剩下的是test,train_percent是trainval中train的百分比，剩下的是val
    # 将train_percent设为1，那么val中则没有数据
    trainval_percent = 1
    train_percent = 0.9
    annotations_is_xml = False
    annotations_is_json = True
    path = r'D:\MyData\data_flydetection\shenyang\voc_seg_shenyang_chedi_data\VOCdevkit2007\VOC2007'
    annotations_path = path + r'\Annotations'
    txtsavepath = path + r'\ImageSets\Main'
    # 读取annotations_path路径下所有文件的名字
    total_annotations = os.listdir(annotations_path)
    num = len(total_annotations)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    trainval_num = len(trainval)
    train_num = len(train)
    val_num = trainval_num - train_num
    test_num = num - trainval_num
    print(f'total_num: {num}, trainval_num: {trainval_num}, test_num: {test_num}, tran_num: {train_num}, val_num: {val_num}')
    ftrainval = open(txtsavepath + r'\trainval.txt', 'w')
    ftest = open(txtsavepath + r'\test.txt', 'w')
    ftrain = open(txtsavepath + r'\train.txt', 'w')
    fval = open(txtsavepath + r'\val.txt', 'w')
    for i in tqdm(list):
        # name = total_annotations[i].split('.')[0] + '\n'
        name = os.path.splitext(total_annotations[i])[0] + '\n'
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
