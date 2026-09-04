"""Fundus preprocessing: crop black border -> resize -> ImageNet-normalize,
plus mild train-only augmentation.

Design notes (for the report):
  * crop_fundus removes the uninformative black margin around the circular
    retina so the model sees mostly retina, and framing is consistent.
  * Val/test transforms are DETERMINISTIC (no augmentation) so evaluation
    is repeatable. Augmentation is applied to TRAIN only.
  * We normalize with ImageNet statistics because our backbone (ResNet50)
    is ImageNet-pretrained.
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_fundus(img_rgb, tol: int = 7):
    """Crop the black border around the circular fundus.

    img_rgb : H x W x 3 uint8 (RGB). Returns the cropped image.
    tol     : pixels brighter than this (in grayscale) count as 'retina'.
              Small value tolerates JPEG noise in the black region.
    """
    if img_rgb.ndim != 3:
        return img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():          # safeguard: all-black image
        return img_rgb
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = img_rgb[y0:y1, x0:x1]
    return cropped if cropped.size else img_rgb


def build_transforms(size: int, train: bool):
    """Return an albumentations pipeline. train=True adds mild augmentation."""
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])