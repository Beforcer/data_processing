from tqdm import tqdm
import os
import random
# 这个程序有个问题：如果有的图片是没有标注信息的，那么就不会写进txt中，需要修改为没有标注的图片放进train里面
if __name__ == '__main__':
    # 将所有数据分为trainval和test，trainval_percent表示trainval的百分比，剩下的是test,train_percent是trainval中train的百分比，剩下的是val
    # 将train_percent设为1，那么val中则没有数据
    trainval_percent = 1
    train_percent = 0.8
    voc_root = r'D:\MyData\voc_chejianluosi_data\VOCdevkit2007'
    voc_version = "VOC2007"
    voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
    voc_xml_path = os.path.join(voc_root, voc_version, "seg")  # Annotations 中是目标检测信息xml,seg中是分割信息json
    txt_save_path = os.path.join(voc_root, voc_version, "ImageSets", "Main")
    # 读取voc_xml_path路径下所有的xml文件的名字
    total_xml = os.listdir(voc_xml_path)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    ftrainval = open(txt_save_path + r'\trainval.txt', 'w')
    ftest = open(txt_save_path + r'\test.txt', 'w')
    ftrain = open(txt_save_path + r'\train.txt', 'w')
    fval = open(txt_save_path + r'\val.txt', 'w')
    for i in tqdm(list):
        # name = total_xml[i][:-4] + '\n'  # 取到倒数第4个字母，即去掉后缀
        name = total_xml[i].split('.')[0] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    total_images_nojpg = []
    total_images = os.listdir(voc_images_path)
    # total_images_nojpg.append(image[:-4] for image in total_images)
    for image in total_images:
        total_images_nojpg.append(image.split('.')[0])
    total_xml_noxml = []
    # total_xml_noxml.append(xml[:-4] for xml in total_xml)
    for xml in total_xml:
        total_xml_noxml.append(xml.split('.')[0])
    num_images = len(total_images)
    if total_images_nojpg != total_xml_noxml:
        for image in total_images_nojpg:
            if image in total_xml_noxml:
                continue
            else:
                image_name = image + '\n'
                ftrain.write(image_name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
