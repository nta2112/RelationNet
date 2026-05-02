# code is based on https://github.com/katerakelly/pytorch-maml
# Extended with JSON split support (2 formats)
#
# ─── JSON Format 1 – Split-list (một file chứa cả 3 splits) ───────────────
#   split.json:
#   {
#     "train": ["n01532829", "n01558993", ...],   ← tên thư mục class
#     "val":   ["n01855672", ...],
#     "test":  ["n01930112", ...]
#   }
#   Ảnh được đọc từ: <data_root>/<class_name>/*.jpg
#
# ─── JSON Format 2 – Image-list (mỗi split một file, kiểu Chen et al.) ───
#   train.json / val.json / test.json:
#   {
#     "label_names": ["class1", "class2", ...],
#     "image_names": ["n01532829/img_001.jpg", ...],   ← path tương đối
#     "image_labels": [0, 0, 1, 1, ...]
#   }
#   Ảnh được đọc từ: <data_root>/<image_name>

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler


# ══════════════════════════════════════════════════════════════════════════════
#  Hàm đọc split
# ══════════════════════════════════════════════════════════════════════════════

def mini_imagenet_folders(train_root='../datas/miniImagenet/train',
                          test_root='../datas/miniImagenet/val'):
    """
    Cách đọc gốc: quét thư mục train/ và val/ (backward compatible).
    """
    metatrain_folders = [
        os.path.join(train_root, label)
        for label in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, label))
    ]
    metatest_folders = [
        os.path.join(test_root, label)
        for label in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, label))
    ]
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)
    return metatrain_folders, metatest_folders


def mini_imagenet_folders_from_split_json(json_path, data_root,
                                          train_key='train', val_key='val', test_key='test'):
    """
    Format 1 – Split-list JSON.
    Returns: (metatrain_folders, metaval_folders, metatest_folders)
    """
    with open(json_path, 'r') as f:
        split_dict = json.load(f)

    def to_abs(key):
        if key not in split_dict:
            return []
        class_names = split_dict[key]
        folders = []
        for cls in class_names:
            path = os.path.join(data_root, cls)
            if not os.path.isdir(path):
                print(f"[WARNING] Folder not found: {path}")
                continue
            folders.append(path)
        return folders

    metatrain_folders = to_abs(train_key)
    metaval_folders   = to_abs(val_key)
    metatest_folders  = to_abs(test_key)

    random.shuffle(metatrain_folders)
    random.shuffle(metaval_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders, metaval_folders, metatest_folders


def mini_imagenet_folders_from_image_json(train_json, test_json, data_root):
    """
    Format 2 – Image-list JSON (kiểu Chen et al. / FEAT).

    Mỗi file JSON chứa:
        {
          "label_names": ["class_A", "class_B", ...],
          "image_names": ["class_A/img001.jpg", "class_A/img002.jpg", ...],
          "image_labels": [0, 0, ...]
        }

    Trả về dict thay vì list thư mục vì ảnh không cần phân thư mục.
    Dùng kết hợp với MiniImagenetTaskFromImageList (xem bên dưới).
    """
    def load(json_path):
        with open(json_path, 'r') as f:
            d = json.load(f)
        required = {'label_names', 'image_names', 'image_labels'}
        missing  = required - set(d.keys())
        if missing:
            raise KeyError(f"JSON thiếu các key: {missing}. File: {json_path}")
        return d

    train_data = load(train_json)
    test_data  = load(test_json)

    def build_class_map(data):
        """Nhóm image paths theo class."""
        class_to_images = {}
        for img_name, label_idx in zip(data['image_names'], data['image_labels']):
            cls = data['label_names'][label_idx]
            if cls not in class_to_images:
                class_to_images[cls] = []
            class_to_images[cls].append(os.path.join(data_root, img_name))
        return class_to_images  # {class_name: [abs_path, ...]}

    metatrain_map = build_class_map(train_data)
    metatest_map  = build_class_map(test_data)
    return metatrain_map, metatest_map


# ══════════════════════════════════════════════════════════════════════════════
#  Task: dùng với Format 1 hoặc folder gốc (list of class folders)
# ══════════════════════════════════════════════════════════════════════════════

class MiniImagenetTask(object):
    """
    Tạo một episode từ danh sách thư mục class.
    Dùng cho mini_imagenet_folders() và mini_imagenet_folders_from_split_json().
    """
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num   = train_num
        self.test_num    = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, range(len(class_folders))))
        samples = {}

        self.train_roots = []
        self.test_roots  = []
        for c in class_folders:
            imgs = [os.path.join(c, x) for x in os.listdir(c)
                    if not x.startswith('.')]
            random.shuffle(imgs)
            samples[c] = imgs
            self.train_roots += samples[c][:train_num]
            self.test_roots  += samples[c][train_num:train_num + test_num]

        self.train_labels = [labels[self._get_class(x)] for x in self.train_roots]
        self.test_labels  = [labels[self._get_class(x)] for x in self.test_roots]

    def _get_class(self, sample):
        # Works on both Linux and Windows paths
        return os.path.dirname(sample)


# ══════════════════════════════════════════════════════════════════════════════
#  Task: dùng với Format 2 (image-list JSON)
# ══════════════════════════════════════════════════════════════════════════════

class MiniImagenetTaskFromImageList(object):
    """
    Tạo một episode từ dict {class_name: [image_paths]}.
    Dùng cho mini_imagenet_folders_from_image_json().
    """
    def __init__(self, class_to_images, num_classes, train_num, test_num):
        self.num_classes = num_classes
        self.train_num   = train_num
        self.test_num    = test_num

        chosen_classes = random.sample(list(class_to_images.keys()), num_classes)
        labels = dict(zip(chosen_classes, range(num_classes)))

        self.train_roots  = []
        self.test_roots   = []
        self.train_labels = []
        self.test_labels  = []

        for cls in chosen_classes:
            imgs = list(class_to_images[cls])
            random.shuffle(imgs)
            self.train_roots  += imgs[:train_num]
            self.test_roots   += imgs[train_num:train_num + test_num]
            self.train_labels += [labels[cls]] * train_num
            self.test_labels  += [labels[cls]] * test_num


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset classes
# ══════════════════════════════════════════════════════════════════════════════

class FewShotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform        = transform
        self.target_transform = target_transform
        self.task             = task
        self.split            = split
        self.image_roots = task.train_roots if split == 'train' else task.test_roots
        self.labels      = task.train_labels if split == 'train' else task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("Abstract class.")


# Global cache to store resized images in RAM
_IMAGE_CACHE = {}

class MiniImagenet(FewShotDataset):
    def __getitem__(self, idx):
        path = self.image_roots[idx]
        if path in _IMAGE_CACHE:
            image = _IMAGE_CACHE[path]
        else:
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            _IMAGE_CACHE[path] = image
            
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


# ══════════════════════════════════════════════════════════════════════════════
#  Sampler
# ══════════════════════════════════════════════════════════════════════════════

class ClassBalancedSampler(Sampler):
    """Samples num_per_class examples each from num_cl classes of size num_inst."""

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl        = num_cl
        self.num_inst      = num_inst
        self.shuffle       = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [
                [i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                for j in range(self.num_cl)
            ]
        else:
            batch = [
                [i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]]
                for j in range(self.num_cl)
            ]
        batch = [item for sublist in batch for item in sublist]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoader helper
# ══════════════════════════════════════════════════════════════════════════════

# Normalization stats (mean/std tính trên mini-ImageNet)
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def get_mini_imagenet_data_loader(task, num_per_class=1, split='train',
                                  shuffle=False, image_size=84):
    """
    Trả về DataLoader cho một episode đã tạo sẵn (task).
    Tương thích với cả MiniImagenetTask và MiniImagenetTaskFromImageList.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        _NORMALIZE,
    ])

    dataset = MiniImagenet(task, split=split, transform=transform)

    num_inst = task.train_num if split == 'train' else task.test_num
    sampler  = ClassBalancedSampler(
        num_per_class, task.num_classes, num_inst, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=num_per_class * task.num_classes,
        sampler=sampler,
        num_workers=0,    # tăng lên nếu cần tốc độ
        pin_memory=True,
    )
