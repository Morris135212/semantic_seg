from PIL import Image
from torch.utils.data import Dataset
import glob
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, scans, ground_truth, transforms=None):
        super(CustomDataset, self).__init__()
        self.scan_file = sorted(glob.glob(scans + '*.png'))
        self.truth_file = sorted(glob.glob(ground_truth+"*.png"))
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
            img, ground_truth = self.transforms((img, ground_truth))
            # try:
            #     img, ground_truth = self.transforms((img, ground_truth))
            # except Exception as e:
            #     print("Could not apply transform.")
            #     print(e)
            #     return
        ground_truth = ground_truth.view((-1, 1))
        return img, ground_truth

    def __len__(self):
        return len(self.truth_file)





