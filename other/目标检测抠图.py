import cv2
import os
from tqdm import tqdm
from lxml import etree
import json

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


if __name__ == '__main__':
    # 该程序将目标检测数据进行抠图并以png的形式保存
    # 这里表示保存抠图之后图片的路径，需要自行更改
    save_path = r'D:\MyData\data_flydetection\chejian\chejian_bolt\voc_cls_chejian_bolt_0320_data\val'
    # 这里的路径表示放mask的路径，自行修改即可
    Annotations_path = r'D:\MyData\data_flydetection\chejian\chejian_bolt\voc_cls_chejian_bolt_0320_data\Annotations'
    image_path = r'D:\MyData\data_flydetection\chejian\chejian_bolt\voc_cls_chejian_bolt_0320_data\JPEGImages'
    class_list = ['bolt1_ok', 'bolt1_ng', 'bolt1_unknown']
    use_json = True
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_files = os.listdir(Annotations_path)
    for file in tqdm(img_files):
        # file = file.split('.')[0]
        file = os.path.splitext(file)[0]
        # img_path = os.path.join(image_path, file + ".png")
        # assert os.path.exists(img_path), "file:{} not exist...".format(img_path)
        if not os.path.exists(os.path.join(image_path, file + ".png")) and os.path.exists(os.path.join(image_path, file + ".jpg")):
            img_path = os.path.join(image_path, file + ".jpg")
        elif not os.path.exists(os.path.join(image_path, file + ".jpg")) and os.path.exists(os.path.join(image_path, file + ".png")):
            img_path = os.path.join(image_path, file + ".png")
        else:
            print(f"image:{file} not exist...")
        # read image
        img = cv2.imread(img_path)
        # 检查xml文件是否存在
        if use_json:
            json_path = os.path.join(Annotations_path, file + ".json")
            assert os.path.exists(json_path), "file:{} not exist...".format(json_path)
            with open(json_path) as fid:
                data = json.load(fid)
            assert "shapes" in data.keys(), "file: '{}' lack of object key.".format(json_path)
            data_object = data["shapes"]
        else:
            xml_path = os.path.join(Annotations_path, file + ".xml")
            assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)
            # read xml
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
            assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
            data_object = data["object"]
        for index, obj in enumerate(data_object):
            # 获取每个object的box信息
            if use_json:
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
            else:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
            if class_name not in class_list and class_name != '1':
                print("{}不在class_list中".format(class_name))
                exit()
            # class_index = class_list.index(class_name)

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                if use_json:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(json_path))
                else:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            crop_image = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            image_name = file + '_' + str(index) + '.png'
            crop_img_path = os.path.join(save_path, f'{class_name}')
            os.makedirs(crop_img_path, exist_ok=True)
            image_file = os.path.join(crop_img_path, image_name)
            cv2.imwrite(image_file, crop_image)