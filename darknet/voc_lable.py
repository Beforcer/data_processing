import xml.etree.ElementTree as ET
import os


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(xmlpath, labelpath, image_id):
    in_file = open(xmlpath + '%s.xml' % image_id)  # 将数据集放于当前目录下
    out_file = open(labelpath + '%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


if __name__ == '__main__':

    sets = ['train', 'test']  # 替换为自己的数据集
    classes = ["2", "4", "5"]  # 修改为自己的类别
    path = r'E:\MyData\shenyangtest_data\VOCdevkit2007\VOC2007'
    labelpath = path + r'\labels\\'
    mainpath = path + r'\ImageSets\Main\\'
    xmlpath = path + r'\Annotations\\'
    if not os.path.exists(labelpath):
        os.makedirs(labelpath)
    for image_set in sets:
        txt_image_ids = open(mainpath + '%s.txt' % image_set)
        image_ids = txt_image_ids.read().strip().split()
        for image_id in image_ids:
            convert_annotation(xmlpath, labelpath, image_id)
    txt_image_ids.close()
