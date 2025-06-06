#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import albumentations as albu
import cv2
import numpy as np
from albumentations import GaussianBlur
# from albumentations.augmentations.blur.transforms import GaussianBlur
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.core.transforms_interface import ImageOnlyTransform


class PepperPatches(ImageOnlyTransform):
    """
    Apply pixel noise to the input image.
    Args:
        value ((float, float, float) or float): color value of the pixel.
        prob (float): probability to add pixels
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        nb_patches=(1, 5),
        height=(0.1, 0.8),
        width=(0.1, 0.8),
        density=(0.05, 0.1),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.density = density
        self.nb_patches = nb_patches
        self.height = height
        self.width = width

    def apply(self, image, **params):
        density = (
            self.density[1] - self.density[0]
        ) * np.random.random_sample() + self.density[0]
        patches = self._get_patches(image.shape[:2])

        for x1, y1, x2, y2 in patches:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if np.random.random_sample() <= density:
                        image[y, x] = 0
        return image

    def _get_patches(self, image_shape):
        image_height, image_width = image_shape[:2]
        offset = int(image_height / 100)
        patches = []
        for _n in range(np.random.randint(*self.nb_patches)):
            patch_height = int(image_height * np.random.uniform(*self.height))
            patch_width = int(image_width * np.random.uniform(*self.width))
            # Offset to ensure the image borders remain white
            y1 = np.random.randint(offset, image_height - patch_height - offset)
            x1 = np.random.randint(offset, image_width - patch_width - offset)
            patches.append((x1, y1, x1 + patch_width, y1 + patch_height))
        return patches


class RandomLines(ImageOnlyTransform):
    """
    Add random lines to the image.
    """

    def __init__(self, nb_lines=(1, 3), thickness=(3, 10), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.nb_lines = nb_lines
        self.thickness = thickness

    def apply(self, image, **params):
        nb_lines = random.randint(*self.nb_lines)
        thickness = random.randint(*self.thickness)

        for i in range(nb_lines):
            x1 = int(random.uniform(0.05, 0.95) * image.shape[0])
            y1 = int(random.uniform(0.05, 0.95) * image.shape[0])
            x2 = int(random.uniform(0.05, 0.95) * image.shape[0])
            y2 = int(random.uniform(0.05, 0.95) * image.shape[0])
            image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=thickness)
        return image


def binarize_image(image, threshold_range=(230, 230), **kwargs):
    threshold = random.randint(*threshold_range)
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


class AddBlackPatch(albu.ImageOnlyTransform):
    def __init__(self, top_left, bottom_right, always_apply=False, p=1.0):
        super(AddBlackPatch, self).__init__(always_apply, p)
        self.top_left = top_left
        self.bottom_right = bottom_right

    def apply(self, image, **params):
        # Unpack the coordinates
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        # Add a black patch to the image at the specified location
        image[y1:y2, x1:x2] = 0
        return image


def get_transforms_dict(augment=False):
    transforms = {}
    transforms_list = []

    if augment:
        transforms_list = []
        transforms_list.append(
            albu.ShiftScaleRotate(
                rotate_limit=0,
                scale_limit=(-0.2, 0),
                shift_limit=(-0.01, 0.01),
                p=0.9,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        transforms_list.append(RandomLines(p=0.05))
        transforms_list.append(PepperPatches(p=0.05))
        transforms_list.append(
            albu.OneOf(
                [
                    albu.Downscale(
                        scale_min=0.95,
                        scale_max=0.95,
                        p=0.5,
                        interpolation={
                            "downscale": cv2.INTER_NEAREST,
                            "upscale": cv2.INTER_NEAREST,
                        },
                    ),
                    GaussianBlur(blur_limit=3, p=0.25),
                ],
                p=0.8,
            )
        )

    transforms["standard"] = albu.Compose(
        transforms_list,
        bbox_params=albu.BboxParams(format="coco", min_area=0, min_visibility=0),
    )

    # Debug
    transforms["debug"] = albu.Compose(
        [], bbox_params=albu.BboxParams(format="coco", min_area=0, min_visibility=0)
    )
    return transforms
