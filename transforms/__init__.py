import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from skimage.color import rgb2gray
from collections import Counter


def calc_sample_weight(label, image):
    image = rgb2gray(image)
    around_texts = cv2.dilate((image < 0.5).astype(np.uint8) | label, np.ones((2, 2)))
    combined = around_texts + label
    # combined = label
    class_counts = np.unique(combined, return_counts=True)[1]
    class_weight = np.sum(class_counts) / class_counts * np.array([1, 2, 2])[:len(class_counts)]
    # class_weight = class_weight / np.max(class_weight)
    weights = np.vectorize(lambda x: class_weight[x])(combined).flatten()
    return weights, class_counts, class_weight


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        if len(data) == 3:
            img, ground, weight = data
        else:
            img, ground = data
        # Extract image as PyTorch tensor
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        try:
            ground = torch.from_numpy(ground)
        except Exception as e:
            ground = None
        if len(data) == 3:
            weight = torch.from_numpy(weight)
            return img, ground, weight
        else:
            if ground is not None:
                return img, ground
            else:
                return img


class ColorJitter:
    def __init__(self, contrast):
        self.contrast = contrast

    def __call__(self, data):
        img, ground = data
        img = transforms.ColorJitter(contrast=self.contrast)(img)
        return img, ground


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.5):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, data):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        img, ground, weight = data
        if random.uniform(0, 1) < self.p:  # 概率的判断
            img_ = img.copy()  # 转化为numpy的形式
            h, w, c = img_.shape  # 获取图像的高，宽，channel的数量
            signal_pct = self.snr  # 设置图像原像素点保存的百分比
            noise_pct = (1 - self.snr)  # 噪声的百分比
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            # random.choice的用法：可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。
            # size表示要输出的numpy的形状
            mask = np.repeat(mask, c, axis=2)  # 将mask在最高轴上（2轴）上复制channel次。
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return img_, ground, weight  # 转化成pil_img的形式
        else:
            return img, ground, weight


class SquarePad:
    def __init__(self, ):
        pass

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, mode='constant', value=0)


class Pad:
    def __init__(self, size=None):
        if size:
            assert len(size) == 2
            self.w, self.h = size
        else:
            self.w = None
            self.h = None

    def __call__(self, data):
        try:
            img, ground = data
        except Exception as e:
            img, ground = data, None
        if not self.w:
            self.w = self.h = max(img.shape[0], img.shape[1])
        img = img / 255
        img = cv2.copyMakeBorder(img,
                                 (self.w - img.shape[0]) // 2,
                                 (self.w - img.shape[0]) - (self.w - img.shape[0]) // 2,
                                 (self.h - img.shape[1]) // 2,
                                 (self.h - img.shape[1]) - (self.h - img.shape[1]) // 2,
                                 cv2.BORDER_REFLECT)
        if ground is not None:
            ground = (255 - ground) / 255
            ground = ground[:, :, 0].astype(np.uint8)
            ground = cv2.copyMakeBorder(ground,
                                        (self.w - ground.shape[0]) // 2,
                                        (self.w - ground.shape[0]) - (self.w - ground.shape[0]) // 2,
                                        (self.h - ground.shape[1]) // 2,
                                        (self.h - ground.shape[1]) - (self.h - ground.shape[1]) // 2,
                                        cv2.BORDER_REFLECT)
        return img, ground


class ToPILImage:
    def __init__(self):
        pass

    def __call__(self, data):
        img, ground = data
        img = transforms.ToPILImage()(img)
        ground = transforms.ToPILImage()(ground)
        return img, ground


class AddNoise:
    def __init__(self, gamma_lower=0.8, gamma_upper=1.25):
        assert gamma_upper > gamma_lower
        self.gamma_lower = gamma_lower
        self.gamma_upper = gamma_upper

    def __call__(self, data):
        img, ground = data
        img = img ** np.random.uniform(self.gamma_lower, self.gamma_upper)
        return img, ground


class Jitter:
    def __init__(self, size=(2302, 1632), jitter_range=50):
        self.size = size
        self.jitter_range = jitter_range

    def __call__(self, data):
        img, ground = data
        jitter = np.random.uniform(0, self.jitter_range, (4, 2)).astype(np.float32)
        pts1 = np.array(((0, 0), (self.size[1], 0), (0, self.size[0]), (self.size[1], self.size[0]))).astype(np.float32)
        pts2 = pts1 + jitter
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (self.size[1], self.size[0]), borderValue=(1, 1, 1))
        ground = cv2.warpPerspective(ground, M, (self.size[1], self.size[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        return img, ground


class AddWeight:
    def __init__(self):
        pass

    def __call__(self, data):
        img, ground = data
        weight, _, _ = calc_sample_weight(ground, img)
        return img, ground, weight


class Resize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, data):
        img, ground = data
        img = cv2.resize(img, (self.w, self.h))
        ground = cv2.resize(ground, (self.w, self.h))
        return img, ground


default_transform = {"train": transforms.Compose([
    Pad(size=(2302, 1632)),
    AddNoise(),
    Jitter(size=(2302, 1632), jitter_range=50),
    AddWeight(),
    # AddPepperNoise(snr=0.98),
    ToTensor()
]),
    "val": transforms.Compose([
        Pad(size=(2302, 1632)),
        AddWeight(),
        ToTensor()
    ]),
    "test": transforms.Compose([
        Pad(size=(2302, 1632)),
        ToTensor()
    ])
}
