"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)
"""
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil
import yaml


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


def translate_info(file_names: list, save_root: str, class_list: list, train_val='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在
        # img_path = os.path.join(voc_images_path, file + ".png")
        if not os.path.exists(os.path.join(voc_images_path, file + ".png")) and os.path.exists(os.path.join(voc_images_path, file + ".jpg")):
            img_path = os.path.join(voc_images_path, file + ".jpg")
        elif not os.path.exists(os.path.join(voc_images_path, file + ".jpg")) and os.path.exists(os.path.join(voc_images_path, file + ".png")):
            img_path = os.path.join(voc_images_path, file + ".png")
        else:
            print(f"image:{file} not exist...")
        # assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查xml文件是否存在
        json_path = os.path.join(voc_json_path, file + ".json")
        assert os.path.exists(json_path), "file:{} not exist...".format(json_path)

        # read xml
        with open(json_path) as fid:
            # xml_str = fid.read()
            data = json.load(fid)
        # xml = etree.fromstring(xml_str)
        # data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["imageHeight"])
        img_width = int(data["imageWidth"])

        # write object info into txt
        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            assert "shapes" in data.keys(), "file: '{}' lack of object key.".format(json_path)
            for index, obj in enumerate(data["shapes"]):
                # 获取每个object的box信息
                if obj['shape_type'] == "rectangle":
                    if obj["points"][0][0] < obj["points"][1][0]:
                        xmin = float(obj["points"][0][0])
                        xmax = float(obj["points"][1][0])
                        ymin = float(obj["points"][0][1])
                        ymax = float(obj["points"][1][1])
                    else:
                        xmax = float(obj["points"][0][0])
                        xmin = float(obj["points"][1][0])
                        ymax = float(obj["points"][0][1])
                        ymin = float(obj["points"][1][1])
                else:
                    continue
                class_name = obj["label"]
                if class_name not in class_list:
                    print("{}不在class_list中".format(class_name))
                    exit()
                class_index = class_list.index(class_name)

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' json, there are some bbox w/h <=0".format(json_path))
                    continue

                # 将box信息转换到yolo格式
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                # info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]
                info = [str(i) for i in [0, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        shutil.copyfile(img_path, os.path.join(save_images_path, img_path.split(os.sep)[-1]))


if __name__ == "__main__":

    # voc数据集根目录以及版本
    voc_root = r"D:\MyData\data_flydetection\shenyang_04\voc_suobi_data\VOCdevkit2007"
    voc_version = "VOC2007"

    # 转换的训练集以及验证集对应txt文件
    train_txt = "train.txt"
    val_txt = "val.txt"

    # 转换后的文件保存目录
    save_file_root = r"D:\MyData\data_flydetection\shenyang_04\yolo_suobi_data"

    # 数据集对应的类别名称，该名称要与yolov5中的yaml文件中的类别一致
    # class_list = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    # class_list = ['zhouxiang', 'goujia', 'fujiakongqishi', 'yixitanhuang', 'kongqitanhuang', 'anquangangsuo', 'gaodufatiaozhenggan', 'anquandiaolian', 'dianxianzhijia', 'suduchuanganqi', 'paizhangqi', 'xiangti']
    # class_list = ['goutou', 'yakuiguan', 'taotongkahuan', 'duizhongzhuangzhi', 'xiangjiaodiangouweizuo', 'lundui', 'tamianzhidongdanyuan', 'zhongxinxiao', 'hengxiangzhidang', 'qianyinlagan', 'lunyuanruihua', 'qianyindianji', 'chilunxiang', 'chilunxiangdiaogan', 'lianzhoujie', 'chilunxiangjiedizhuangzhi', 'chaxuntianxian', 'kongyaji', 'xiangti']
    # class_list = ['suo_ok', 'suo_ng']
    class_list = ['suobi_ok', 'suobi_ng']
    # class_list = ['suoba_ok', 'suoba_ng']
    # class_list = ['task']
    # class_list = ['0', '1', '2', 'bolt1_ok', 'bolt1_ng', 'bolt1_unknown', 'OK', 'NG', 'Unknown'] # 0是OK，1是NG，2是unknown, bolt1_unknow需要修改为bolt1_unknown
    # class_list = ['abnormal']
    # class_list = ['corner', 'luoshuan']
    # class_list = ['wentie_ok', 'wentie_ng']
    # class_list = ['gongyidu_ok', 'gongyidu_ng']
    # class_list = ['FOD', 'abnormal']
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
    translate_info(train_file_names, save_file_root, class_list, "train")

    # 读取val.txt中的所有行信息，删除空行
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(val_file_names, save_file_root, class_list, "val")
    # 生成yaml文件
    config_yaml = {"path": save_file_root, "train": 'train', "val": 'val', "nc": len(class_list), "names": class_list}
    config_path = os.path.join(save_file_root, 'config.yaml')
    with open(config_path, "w") as f:  # 写文件
        yaml.safe_dump(data=config_yaml, stream=f, indent=4, sort_keys=False, encoding='utf-8')

