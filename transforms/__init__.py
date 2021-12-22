import torch
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
