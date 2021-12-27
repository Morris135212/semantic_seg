from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
import glob
import numpy as np

import transforms


class CustomDataset(Dataset):
    def __init__(self, scans, ground_truth, transforms=None):
        super(CustomDataset, self).__init__()
        # self.scan_file = sorted(glob.glob(scans + '*.png'))
        self.scan_file = scans
        # self.truth_file = sorted(glob.glob(ground_truth+"*.png"))
        self.truth_file = ground_truth
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.scan_file[index % len(self.scan_file)].rstrip()
        try:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Can not read img_path {img_path}")
            return
        truth_path = self.truth_file[index % len(self.truth_file)].rstrip()
        try:
            ground_truth = np.array(Image.open(truth_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Can not read Ground truth {truth_path}")
            return
        if self.transforms:
            data = self.transforms((img, ground_truth))
            if len(data) == 3:
                img, ground_truth, weight = data
            else:
                img, ground_truth = data
            # try:
            #     img, ground_truth = self.transforms((img, ground_truth))
            # except Exception as e:
            #     print("Could not apply transform.")
            #     print(e)
            #     return
        ground_truth = ground_truth.view((-1,))
        if len(data) == 3:
            return img, ground_truth, weight
        else:
            return img, ground_truth

    def __len__(self):
        return len(self.truth_file)


class TestDataset(Dataset):
    def __init__(self, scans, transforms=transforms.default_transform["val"]):
        super(TestDataset, self).__init__()
        # self.scan_file = sorted(glob.glob(scans + '*.png'))
        self.scan_file = scans
        # self.truth_file = sorted(glob.glob(ground_truth+"*.png"))
        self.transforms = transforms

    def __len__(self):
        return len(self.scan_file)

    def __getitem__(self, index):
        img_path = self.scan_file[index % len(self.scan_file)].rstrip()
        try:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Can not read img_path {img_path}")
            return
        if self.transforms:
            data = self.transforms(img)
        return data


