import random
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, ground = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        ground = transforms.ToTensor()(ground)

        return img, ground


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

    def __init__(self, snr, p=0.9):
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
        img, ground = data
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
            return img_, ground  # 转化成pil_img的形式
        else:
            return img, ground


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
    def __init__(self, size):
        assert len(size) != 2
        self.w, self.h = size

    def __call__(self, data):
        img, ground = data
        img = img / 255
        img = cv2.copyMakeBorder(img,
                                 (self.w - img.shape[0]) // 2,
                                 (self.w - img.shape[0]) - (self.w - img.shape[0]) // 2,
                                 (self.h - img.shape[1]) // 2,
                                 (self.h - img.shape[1]) - (self.h - img.shape[1]) // 2,
                                 cv2.BORDER_REFLECT)

        ground = (255 - ground) / 255
        ground = ground[:, :, 0].astype(np.uint8)
        ground = cv2.copyMakeBorder(ground,
                                    (self.w - ground.shape[0]) // 2,
                                    (self.w - ground.shape[0]) - (self.w - ground.shape[0]) // 2,
                                    (self.h - ground.shape[1]) // 2,
                                    (self.h - ground.shape[1]) - (self.h - ground.shape[1]) // 2,
                                    cv2.BORDER_REFLECT)
        return img, ground


default_transform = {"train": transforms.Compose([
    Pad((2302, 1632)),
    ColorJitter(0.5),
    AddPepperNoise(snr=0.7),
    ToTensor()
]),
    "val": transforms.Compose([
        Pad((2302, 1632)),
        ToTensor()
    ])}