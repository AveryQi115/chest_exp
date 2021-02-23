import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, sampler
from torch.utils.data.sampler import BatchSampler

import torch
import torch.distributed as dist
from DistributedProxySampler import DistributedProxySampler
from PIL import Image
import random, cv2
import os
import numpy as np
import json

class ChestRayCFG():
    def __init__(self):
        self.INPUT_SIZE = (320,320)
        self.DATASET_TRAIN_JSON = "../accv_cls/ChestRayNIH-train-10.json"
        self.DATASET_VALID_JSON = "../accv_cls/ChestRayNIH-test.json"
        self.DATASET_TEST_JSON = "../accv_cls/ChestRayNIH-test.json"

class CheXpertCFG():
    def __init__(self):
        self.INPUT_SIZE = (320,320)
        self.DATASET_UNCERTAIN = "U-positive"
        self.DATASET_TRAIN_JSON = "chexpert_moco_clean_10%.json"
        self.DATASET_UNLABELED_JSON = "chexpert_moco_clean_90%.json"
        self.DATASET_VALID_JSON = "chexpert_moco_valid.json"
        self.DATASET_TEST_JSON = "chexpert_moco_clean_test.json"

class CheXpert(Dataset):
    def __init__(self, mode='train', transform=None,cfg=CheXpertCFG(),fixmatch=False,weak_transform=None):
        super().__init__()
        random.seed(0)
        self.mode = mode
        self.transform = transform
        self.weak_transform = weak_transform
        self.fixmatch = fixmatch
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
        elif "unlabeled" in self.mode:
            self.json_path = cfg.DATASET_UNLABELED_JSON

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
        if not ('unlabeled' in self.mode):
            image = self.transform(img)
            label = self._get_label(now_info)
            return image, label
        else:
            image_s = self.transform(img)
            image_w = self.weak_transform(img)
            label = self._get_label(now_info)
            return image_s,image_w,label



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


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__ 
                      if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)

def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 4,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)
        
        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1
        
        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset)*num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)
        
        if data_sampler.__name__ == 'RandomSampler':    
            data_sampler = data_sampler(dset, replacement, int(num_samples), generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")
        
        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, int(batch_size), drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, 
                          num_workers=num_workers, pin_memory=pin_memory)