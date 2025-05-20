# import argparse

import skimage.transform as trans
import numpy as np

import SimpleITK as sitk
# import cv2


import torch.optim as optim
import math

class Optimizer:

    def get_optimizer(self, config_optim, parameters):
        
        if config_optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                            betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                            eps=config_optim.eps)
        elif config_optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
        elif config_optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=config_optim.lr, weight_decay=1e-4, momentum=0.9)
        else:
            raise NotImplementedError(
                'Optimizer {} not understood.'.format(config_optim.optimizer))

    def adjust_learning_rate(self, optimizer, epoch, config):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < config.training.warmup_epochs:
            lr = config.optim.lr * epoch / config.training.warmup_epochs
        else:
            lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - config.training.warmup_epochs) / (
                        config.training.epochs - config.training.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr


def resample_3D_nii_to_Fixed_size(nii_image, image_new_size, resample_methold=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    image_original_size = nii_image.GetSize()  # 原始图像的尺寸
    image_original_spacing = nii_image.GetSpacing()  # 原始图像的像素之间的距离
    image_new_size = np.array(image_new_size, float)
    factor = image_original_size / image_new_size
    image_new_spacing = image_original_spacing * factor
    image_new_size = image_new_size.astype(np.int)

    resampler.SetReferenceImage(nii_image)  # 需要resize的图像（原始图像）
    resampler.SetSize(image_new_size.tolist())
    resampler.SetOutputSpacing(image_new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_methold)

    return resampler.Execute(nii_image)


def nii_resize_2D(image, label, shape):
    """
    type of image,label: Image or array or None

    :return: array or None
    """
    # image
    if isinstance(image, sitk.SimpleITK.Image):  # image need type array, if not, transform it
        image = sitk.GetArrayFromImage(image)
    if image is not None:
        image = trans.resize(image, (shape, shape))
    # label
    if isinstance(label, np.ndarray):
        label = sitk.GetImageFromArray(label)  # label1 need type Image
    if label is not None:
        label = resample_3D_nii_to_Fixed_size(label, (shape, shape),
                                              resample_methold=sitk.sitkNearestNeighbor)
        label = sitk.GetArrayFromImage(label)
    return image, label

