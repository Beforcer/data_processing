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
import numpy as np
import yaml


def translate_info(file_names: list, save_root: str, class_list: list, train_val='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names: ImageSets中的train.txt或者val.txt中的图像名称列表
    :param save_root: 保存yolo格式的标志格式和图像的路径
    :param class_dict: 标注的label的list
    :param train_val: 转换train或者val
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
        img_path = os.path.join(voc_images_path, file + ".png")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查json文件是否存在
        json_path = os.path.join(voc_json_path, file + ".json")
        # assert os.path.exists(json_path), "file:{} not exist...".format(json_path)

        if os.path.exists(json_path):
            with open(json_path) as fid:
                json_info = json.load(fid)
            img_height = json_info["imageHeight"]
            img_width = json_info["imageWidth"]
            np_w_h = np.array([[img_width, img_height]], np.int32)

            with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
                # 每一个分割区域算一个实例
                for point_json in json_info["shapes"]:
                    if point_json["shape_type"] == "polygon":
                        txt_content = ""
                        np_points = np.array(point_json["points"], np.int32)

                        class_name = point_json["label"]
                        # class_name = obj["name"]
                        if class_name not in class_list:
                            print(f"{class_name}不在class_list中,该图像名称是{file}")
                            exit()
                        class_index = class_list.index(class_name)
                        norm_points = np.around(np_points / np_w_h, 6)
                        # norm_points = np_points / np_w_h, 6
                        norm_points_list = norm_points.tolist()
                        txt_content += str(class_index) + " " + " ".join(
                            [" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
                        f.write(txt_content)
                # 多个分割区域可能是一个实例，下面的代码是一个图中就一个实例，多个分割区域合并
                # seg_points_list = []
                # for point_json in json_info["shapes"]:
                #     if point_json["shape_type"] == "polygon":
                #         txt_content = ""
                #         np_points = np.array(point_json["points"], np.int32)
                #
                #         class_name = point_json["label"]
                #         # class_name = obj["name"]
                #         if class_name not in class_list:
                #             print(f"{class_name}不在class_list中,该图像名称是{file}")
                #             exit()
                #         class_index = class_list.index(class_name)
                #         norm_points = np.around(np_points / np_w_h, 6)
                #         # norm_points = np_points / np_w_h, 6
                #         norm_points_list = norm_points.tolist()
                #         seg_points_list.extend(norm_points_list)
                # txt_content += str(class_index) + " " + " ".join(
                #     [" ".join([str(cell[0]), str(cell[1])]) for cell in seg_points_list]) + "\n"
                # f.write(txt_content)
        shutil.copyfile(img_path, os.path.join(save_images_path, img_path.split(os.sep)[-1]))


if __name__ == "__main__":
    # 该程序将labelme标注的分割数据转换为yolo数据
    # voc数据集根目录以及版本
    voc_root = r"D:\MyData\data_flydetection\shenyang\voc_seg_shenyang_chedi_data\VOCdevkit2007"
    voc_version = "VOC2007"

    # 转换的训练集以及验证集对应txt文件
    train_txt = "train.txt"
    val_txt = "val.txt"

    # 转换后的文件保存目录
    save_file_root = r"D:\MyData\data_flydetection\shenyang\yolo_seg_shenyang_chedi_data"

    # 数据集对应的类别名称，该名称要与yolov5中的yaml文件中的类别一致
    # 车间车侧
    # class_list = ['yixitanhuang', 'duangai', 'zhouxiang', 'dianxianzhijia', 'goujia', 'fujiakongqishi', 'zhawa',
    #               'lundui', 'tamianzhidongdanyuan', 'zhongxinxiao', 'paizhangqi', 'qianyinlagan']
    # 车间车底
    # class_list = ['lundui', 'qianyinlagan', 'zhongxinxiao', 'chayafa', 'tamianzhidongdanyuan', 'hengxiangzhidang']
    # 沈阳车侧
    # class_list = ['goujia', 'duangai', 'yixitanhuang', 'zhouxiang', 'suduchuanganqi', 'dianxianzhijia', 'anquangangsuo',
    #               'fujiakongqishi', 'zhawa', 'kongqitanhuang', 'gaodufatiaozhenggan', 'anquandiaolian', 'chufenggang',
    #               'fuzhukongzhiban', 'kongyaji', 'TMS_A', 'xishuiwufaxiang', 'TRB', 'jiedikaiguanxiang',
    #               'zhinengdiancifa',
    #               'xianludiankangqi', 'IVS', 'MS/DS', 'TMS_B', 'EXB', 'shuixiang', 'miehuoxitong', 'zhidongdianzuxiang',
    #               'HB', 'VVVF', 'yixitanghuang', 'gaoyagongdianxiang', 'SPS', 'SIV', 'xudianchixiang', 'paizhangqi',
    #               'xiangti', 'kongqitanhang']
    # 沈阳车底
    class_list = ['lundui', 'zhongxinxiao', 'qianyinlagan', 'chayafa', 'hengxiangzhidang', 'tamianzhidongdanyuan',
                  'zhongxinixao', 'chegouyouzui', 'chegougoushe', 'paizhangqi', 'taotongkahuan', 'yakuiguan',
                  'gouweizuo', 'goutou', 'goushe', 'youzui', 'banzidongchegoutaotongkahuan',
                  'banzidongchegoutaotongkahuanhouduan',
                  'duizhongzhuangzhi', 'banzidongchegougouweizuo', 'lunyuanrunhua', 'zhuanxiangjiachayafa',
                  'chaxuntianxian', 'chufenggang', 'xiangti', 'chufenggangdiaozhuang', 'kongyaji',
                  'banyongjiuchegoutaotongkahuan',
                  'banyongjiuchegougouweizuo', 'banyongjiuchegou', 'chilunxiang', 'qianyindianji',
                  'chilunxiangjiedizhuangzhi',
                  'lianzhoujie', 'danqiguan', 'shuiguan']
    # class_list = ['zhawa']
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
