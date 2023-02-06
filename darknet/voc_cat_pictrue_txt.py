
if __name__ == '__main__':
    path_write_string = r'E:/MyData/shenyangtest_data/VOCdevkit2007/VOC2007/JPEGImages/'
    path = r'E:\MyData\shenyangtest_data\VOCdevkit2007\VOC2007\\'
    txt_train_read = open(path + r'\ImageSets\Main\train.txt', 'r')
    txt_test_read = open(path + r'\ImageSets\Main\test.txt', 'r')
    txt_trainpath_write = open(path + r'\2007_train.txt', 'w')
    txt_testpath_write = open(path + r'\2007_test.txt', 'w')
    txt_train_data = txt_train_read.read().splitlines()
    txt_test_data = txt_test_read.read().splitlines()
    num = len(txt_train_data)
    for name in txt_train_data:
        txt_trainpath_write.write(path_write_string + name + '.jpg' + '\n')
    for name in txt_test_data:
        txt_testpath_write.write(path_write_string + name + '.jpg' + '\n')
    txt_trainpath_write.close()
    txt_testpath_write.close()
