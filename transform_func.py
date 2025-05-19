from monai import transforms, data
import numpy as np
import torch


def custom_collate(batch_list):
    # batch_list 是一个 list，每个元素是一个字典：{'image': tensor, 'path': str}
    images = [item['image'] for item in batch_list]
    paths = [item['path'] for item in batch_list]

    # 假设每个 image 是 [C, H, W, D]，我们先 stack 成 [B, C, H, W, D]
    image = torch.stack(images, dim=0)

    # 开始你的处理
    image = image.permute(0, 1, 4, 2, 3)      # [B, C, H, W, D] → [B, C, D, H, W]
    B, C, D, H, W = image.shape
    image = image.view(B * C, D, H, W)        # → [B*C, D, H, W]
    image = image.unsqueeze(1)               # → [B*C, 1, D, H, W]
    image = image.repeat(1, 3, 1, 1, 1)       # → [B*C, 3, D, H, W]

    return {"image": image, "path": paths}

train_transforms = transforms.Compose(
    [
        transforms.CopyItemsd(keys=["image"], names=["path"]),
        transforms.LoadImaged(keys=["image","brainmask"]),
        transforms.EnsureChannelFirstd(keys=["image","brainmask"]),
        transforms.EnsureTyped(keys=["image","brainmask"]),
        transforms.Orientationd(keys=["image","brainmask"], axcodes="RAS"),
        transforms.RandAffined(
            keys=["image","brainmask"],
            rotate_range=(-np.pi / 36, np.pi / 36),
            translate_range=(-1, 1),
            scale_range=(-0.05, 0.05),
            padding_mode="zeros",
            prob=0.5,
        ),
        transforms.CropForegroundd(
            keys=["image", "brainmask"],
            source_key="brainmask",
            allow_smaller=False,
        ),
        transforms.ResizeWithPadOrCropd(keys=["image","brainmask"], spatial_size=(192, 192, 141)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0, b_max=1,clip=True ),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.CopyItemsd(keys=["image"], names=["path"]),
        transforms.LoadImaged(keys=["image","brainmask"]),
        transforms.EnsureChannelFirstd(keys=["image","brainmask"]),
        transforms.EnsureTyped(keys=["image","brainmask"]),
        transforms.Orientationd(keys=["image","brainmask"], axcodes="RAS"),
        transforms.CropForegroundd(
            keys=["image", "brainmask"],
            source_key="brainmask",
            allow_smaller=False,
        ),
        transforms.ResizeWithPadOrCropd(keys=["image","brainmask"], spatial_size=(192, 192, 141)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0, b_max=1),
    ]
)