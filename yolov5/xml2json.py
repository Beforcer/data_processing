import copy
import os
from tqdm import tqdm
from lxml import etree
import json

json_data = {
    "version": "0.2.23",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": 1024,
    "imageWidth": 1024,
    "text": ""

}
object = {
    "label": "Unknown",
    "text": "",
    "points": [],
    "group_id": None,
    "shape_type": "rectangle",
    "flags": {}

}


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, train_val='xml'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    # save_txt_path = os.path.join(save_root, train_val, "labels")
    # if os.path.exists(save_txt_path) is False:
    #     os.makedirs(save_txt_path)
    # save_images_path = os.path.join(save_root, train_val, "images")
    # if os.path.exists(save_images_path) is False:
    #     os.makedirs(save_images_path)
    labe_list = []
    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在
        # 检查xml文件是否存在
        xml_path = os.path.join(voc_xml_path, file)
        assert os.path.exists(xml_path), "file:{} not exist...".format(train_val)
        json_info = copy.deepcopy(json_data)
        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])
        img_name = data["filename"]
        json_info["imageHeight"] = img_height
        json_info["imageWidth"] = img_width
        json_info["imagePath"] = os.path.join('..', 'JPEGImages', img_name)
        assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        for index, obj in enumerate(data["object"]):
            # 获取每个object的box信息
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                exit()
            class_name = obj["name"]
            object_info = copy.deepcopy(object)
            object_info["label"] = class_name
            object_info["shape_type"] = "rectangle"
            object_info["points"] = [[xmin, ymin], [xmax, ymax]]
            json_info["shapes"].append(object_info)
            if class_name not in labe_list:
                labe_list.append(class_name)
        json.dumps(json_info, indent=4)
        filename = os.path.join(save_root, os.path.splitext(file)[0] + '.json')
        with open(filename, 'w') as file_obj:
            json.dump(json_info, file_obj, indent=4)
    print(labe_list)


if __name__ == "__main__":

    # voc数据集根目录以及版本
    voc_root = r"D:\MyData\voc_det_chejian_FOD_data\VOCdevkit2007"
    voc_version = "VOC2007"

    # 转换的训练集以及验证集对应txt文件
    # train_txt = "train.txt"
    # val_txt = "val.txt"

    # 转换后的文件保存目录
    save_file_root = os.path.join(voc_root, voc_version, "Annotations_json")
    os.makedirs(save_file_root, exist_ok=True)
    # 拼接出voc的images目录，xml目录，txt目录
    # voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
    voc_xml_path = os.path.join(voc_root, voc_version, "Annotations")
    # train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
    # val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

    # 检查文件/文件夹都是否存在
    # assert os.path.exists(voc_images_path), "VOC images path not exist..."
    assert os.path.exists(voc_xml_path), "VOC xml path not exist..."
    # assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
    # assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
    if os.path.exists(save_file_root) is False:
        os.makedirs(save_file_root)

    # 读取train.txt中的所有行信息，删除空行
    # with open(train_txt_path, "r") as r:
    #     train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    train_file_names = []
    for file in os.listdir(voc_xml_path):
        if os.path.splitext(file)[1] == '.xml':
            train_file_names.append(file)
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(train_file_names, save_file_root, "xml")