import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import random, cv2
import os
import numpy as np
import json

class ChestRayCFG():
    def __init__(self):
        self.INPUT_SIZE = (320,320)
        self.DATASET_TRAIN_JSON = "../accv_cls/ChestRayNIH-train.json"
        self.DATASET_VALID_JSON = "../accv_cls/ChestRayNIH-test.json"
        self.DATASET_TEST_JSON = "../accv_cls/ChestRayNIH-test.json"

class CheXpertCFG():
    def __init__(self):
        self.INPUT_SIZE = (320,320)
        self.DATASET_UNCERTAIN = "U-positive"
        self.DATASET_TRAIN_JSON = "chexpert_moco_clean.json"
        self.DATASET_VALID_JSON = "chexpert_moco_valid.json"
        self.DATASET_TEST_JSON = "chexpert_moco_clean_test.json"


class CheXpert(Dataset):
    def __init__(self, mode='train', transform=None,cfg=CheXpertCFG()):
        super().__init__()
        random.seed(0)
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE

        if self.mode == "train":
            print("Loading train data ...")
            self.json_path = cfg.DATASET_TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...")
            self.json_path = cfg.DATASET_VALID_JSON
        elif "test" in self.mode:
            print("Loading test data ...")
            self.json_path = cfg.DATASET_TEST_JSON

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]
        self.data = self.all_info["annotations"]
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))
        
        transform_uncertain = [{"nan":0,0:0,1:1,-1:0},
                                {"nan":0,0:0,1:1,-1:1},
                                {"nan":0,0:0,1:1,-1:-1}]
        if cfg.DATASET_UNCERTAIN == "U-positive":
            self.transform_dict = transform_uncertain[1]
        elif cfg.DATASET_UNCERTAIN == "U-negative":
            self.transform_dict == transform_uncertain[0]
        elif cfg.DATASET_UNCERTAIN == "U-ignore":
            self.transform_dict == transform_uncertain[2]


    def __getitem__(self, index):
        now_info = self.data[index]
        img = Image.open(now_info['path'])
        image = self.transform(img)
        label = self._get_label(now_info)
        return image, label


    def _get_label(self,now_info):
        label = []
        for key in now_info.keys():
            if key in ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']:
                label.append(self.transform_dict[now_info[key]])
        label = np.array(label)
        assert label.shape == (self.num_classes,)
        return label


    def get_annotations(self):
        clean_anno = []
        for anno in self.data:
            clean_anno_dict=dict()
            for key in anno.keys():
                if key not in ['path','Sex','Age','Frontal/Lateral','AP/PA']:
                    clean_anno_dict[key] = self.transform_dict[anno[key]]
            clean_anno.append(clean_anno_dict)
        return clean_anno

    def get_num_classes(self):
        return self.num_classes

    def __len__(self):
        return len(self.data)

class ChestRay(Dataset):
    def __init__(self, mode='train', transform=None,cfg=ChestRayCFG()):
        super().__init__()
        random.seed(0)
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE

        if self.mode == "train":
            print("Loading train data ...")
            self.json_path = cfg.DATASET_TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...")
            self.json_path = cfg.DATASET_VALID_JSON
        elif "test" in self.mode:
            print("Loading test data ...")
            self.json_path = cfg.DATASET_TEST_JSON

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]
        self.data = self.all_info["annotations"]
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))


    def __getitem__(self, index):
        now_info = self.data[index]
        img = Image.open(now_info['path'])
        image = self.transform(img)
        label = self._get_label(now_info)
        return image, label


    def _get_label(self,now_info):
        label = []
        for key in now_info.keys():
            if key not in ['path']:
                label.append(now_info[key])
        label = np.array(label)
        assert label.shape == (self.num_classes,)
        return label


    def get_annotations(self):
        clean_anno = []
        for anno in self.data:
            clean_anno_dict=dict()
            for key in anno.keys():
                if key not in ['path']:
                    clean_anno_dict[key] = anno[key]
            clean_anno.append(clean_anno_dict)
        return clean_anno

    def get_num_classes(self):
        return self.num_classes

    def __len__(self):
        return len(self.data)
