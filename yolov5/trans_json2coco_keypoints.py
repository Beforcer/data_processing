"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)
"""
import os
from tqdm import tqdm
import json
import shutil
import yaml
from pycocotools.coco import COCO


def process_single_json(labelme, image_id=1, bbox_id=0):
    '''
    输入labelme的json数据，输出coco格式的每个框的关键点标注信息
    '''

    coco_annotations = []

    for each_ann in labelme['shapes']:  # 遍历该json文件中的所有标注

        if each_ann['shape_type'] == 'rectangle' and each_ann['label'] == "luoshuan":  # 筛选出个体框

            # 个体框元数据
            bbox_dict = {}
            bbox_dict['category_id'] = 1
            bbox_dict['segmentation'] = []

            bbox_dict['iscrowd'] = 0
            bbox_dict['segmentation'] = []
            bbox_dict['image_id'] = image_id
            bbox_dict['id'] = bbox_id

            # 获取个体框坐标
            bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
            bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
            bbox_w = bbox_right_bottom_x - bbox_left_top_x
            bbox_h = bbox_right_bottom_y - bbox_left_top_y
            bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]  # 左上角x、y、框的w、h
            bbox_dict['area'] = bbox_w * bbox_h

            # 筛选出分割多段线
            for each_ann in labelme['shapes']:  # 遍历所有标注
                if each_ann['shape_type'] == 'polygon':  # 筛选出分割多段线标注
                    # 第一个点的坐标
                    first_x = each_ann['points'][0][0]
                    first_y = each_ann['points'][0][1]
                    if (first_x > bbox_left_top_x) & (first_x < bbox_right_bottom_x) & (
                            first_y < bbox_right_bottom_y) & (first_y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        bbox_dict['segmentation'] = list(
                            map(lambda x: list(map(lambda y: round(y, 2), x)), each_ann['points']))  # 坐标保留两位小数
                        # bbox_dict['segmentation'] = each_ann['points']

            # 筛选出该个体框中的所有关键点
            bbox_keypoints_dict = {}
            for each_ann in labelme['shapes']:  # 遍历所有标注

                if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                    # 关键点横纵坐标
                    x = int(each_ann['points'][0][0])
                    y = int(each_ann['points'][0][1])
                    label = each_ann['label']
                    if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & (y < bbox_right_bottom_y) & (
                            y > bbox_left_top_y):  # 筛选出在该个体框中的关键点
                        bbox_keypoints_dict[label] = [x, y]

            bbox_dict['num_keypoints'] = len(bbox_keypoints_dict)
            # print(bbox_keypoints_dict)

            # 把关键点按照类别顺序排好
            bbox_dict['keypoints'] = []
            for each_class in class_list['keypoints']:
                if each_class in bbox_keypoints_dict:
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])
                    bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])
                    bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
                else:  # 不存在的点，一律为0
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)
                    bbox_dict['keypoints'].append(0)

            coco_annotations.append(bbox_dict)

    return coco_annotations


def translate_info(file_names: list, save_root: str, train_val='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param keypoints_list:
    :param train_val:
    :return:
    """
    IMG_ID = 0
    ANN_ID = 0

    coco = {}

    coco['categories'] = []
    coco['categories'].append(class_list)

    coco['images'] = []
    coco['annotations'] = []

    save_coco_path = save_root
    if os.path.exists(save_coco_path) is False:
        os.makedirs(save_coco_path)
    save_images_path = os.path.join(save_root, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在
        img_path = os.path.join(voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查json文件是否存在
        json_path = os.path.join(voc_json_path, file + ".json")
        assert os.path.exists(json_path), "file:{} not exist...".format(json_path)

        # read json
        with open(json_path) as fid:
            # xml_str = fid.read()
            data = json.load(fid)
        # 提取图像元数据
        img_dict = {}
        img_dict['file_name'] = data['imagePath']
        img_dict['height'] = data['imageHeight']
        img_dict['width'] = data['imageWidth']
        img_dict['id'] = IMG_ID
        coco['images'].append(img_dict)

        # img_height = int(data["imageHeight"])
        # img_width = int(data["imageWidth"])

        # write object info into coco
        # 提取框和关键点信息
        coco_annotations = process_single_json(data, image_id=IMG_ID, bbox_id=ANN_ID)
        coco['annotations'] += coco_annotations
        IMG_ID += 1
        ANN_ID += 1
        with open(os.path.join(save_coco_path, f'{train_val}_coco.json'), 'w') as f:
            json.dump(coco, f, indent=2)
        # copy image into save_images_path
        shutil.copyfile(img_path, os.path.join(save_images_path, img_path.split(os.sep)[-1]))
    print(f"验证{train_val}集coco格式")
    my_coco = COCO(os.path.join(save_file_root, f'{train_val}_coco.json'))


if __name__ == "__main__":

    # voc数据集根目录以及版本
    voc_root = r"D:\MyData\data_flydetection\chejian_demo\voc_keypoints_luosi_data\VOCdevkit2007"
    voc_version = "VOC2007"

    # 转换的训练集以及验证集对应txt文件
    train_txt = "train.txt"
    val_txt = "val.txt"

    # 转换后的文件保存目录
    save_file_root = r"D:\MyData\data_flydetection\chejian_demo\mm_keypointsv3_lushuan_data"

    # 数据集对应的类别名称，该名称要与yolov5中的yaml文件中的类别一致
    class_list = {
        'supercategory': 'luoshuan',
        'id': 1,
        'name': 'luoshuan',
        'keypoints': ['1', '2', '3', '4', '5'],  # 大小写敏感
        'skeleton': [[0, 1], [1, 2], [2, 3], [3, 4]]
    }

    # 拼接出voc的images目录，xml目录，txt目录
    voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
    voc_json_path = os.path.join(voc_root, voc_version, "Annotations")
    train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
    val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

    # 检查文件/文件夹都是否存在
    assert os.path.exists(voc_images_path), "VOC images path not exist..."
    assert os.path.exists(voc_json_path), "VOC json path not exist..."
    assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
    assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
    if os.path.exists(save_file_root) is False:
        os.makedirs(save_file_root)

    # 读取train.txt中的所有行信息，删除空行
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(train_file_names, save_file_root, "train")

    # 读取val.txt中的所有行信息，删除空行
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(val_file_names, save_file_root, "val")

