import json
import os
from tqdm import tqdm

if __name__ == "__main__":
    json_path = r'D:\MyData\data_flydetection\chejian\chejian_bolt\voc_det_chejian_bolt20240111_data\VOCdevkit2007\VOC2007\Annotations'
    assert os.path.exists(json_path), "VOC json path not exist..."
    # old_label = ['boilt1_unknown']
    # new_lable = 'bolt1_unknown'
    label_list = []
    json_list = os.listdir(json_path)
    for file in tqdm(json_list):
        path = os.path.join(json_path, file)
        with open(path) as fid:
            json_data = json.load(fid)
        for index, obj in enumerate(json_data["shapes"]):
            # if obj['shape_type'] in ['rectangle'] and obj["label"] in old_label:
            #     obj["label"] = new_lable
            # if obj['shape_type'] in ['rectangle'] and obj["label"] == 'boilt1_ok':
            #     obj["label"] = 'bolt1_ok'
            # if obj['shape_type'] in ['rectangle'] and obj["label"] == 'boilt1_ng':
            #     obj["label"] = 'bolt1_ng'
            if obj['shape_type'] in ['rectangle'] and obj["label"] == 'bolt1_unknow':
                obj["label"] = 'bolt1_unknown'
            if obj["label"] not in label_list:
                label_list.append(obj["label"])
                # print(file)
        json.dumps(json_data, indent=4)
        with open(path, 'w') as fid:
            json.dump(json_data, fid, indent=4)
    print(label_list)