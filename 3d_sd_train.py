import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset, DecathlonDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

set_determinism(42)
root_dir = 'ldm'

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(2, 2, 2), mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(80, 96, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=(-np.pi / 36, np.pi / 36),
            translate_range=(-1, 1),
            scale_range=(-0.05, 0.05),
            padding_mode="zeros",
            prob=0.5,
        ),
        transforms.CopyItemsd(keys=["image"], times=4, names=["flair","t1","t1c",'t2']),
        transforms.Lambdad(keys="flair", func=lambda x: x[0, :, :, :]),
        transforms.Lambdad(keys="t1", func=lambda x: x[1, :, :, :]),
        transforms.Lambdad(keys="t1c", func=lambda x: x[2, :, :, :]),
        transforms.Lambdad(keys="t2", func=lambda x: x[3, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["flair"], channel_dim="no_channel"),
        transforms.EnsureChannelFirstd(keys=["t1"], channel_dim="no_channel"),
        transforms.EnsureChannelFirstd(keys=["t1c"], channel_dim="no_channel"),
        transforms.EnsureChannelFirstd(keys=["t2"], channel_dim="no_channel"),
    ]
)
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",  # validation
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=8,
    download=True,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=train_transforms,
)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True)

check_data = first(train_loader)

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(2, 2, 2), mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(80, 96, 64)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.CopyItemsd(keys=["image"], times=4, names=["flair","t1","t1c",'t2']),
        transforms.Lambdad(keys="flair", func=lambda x: x[0, :, :, :]),
        transforms.Lambdad(keys="t1", func=lambda x: x[1, :, :, :]),
        transforms.Lambdad(keys="t1c", func=lambda x: x[2, :, :, :]),
        transforms.Lambdad(keys="t2", func=lambda x: x[3, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["flair"], channel_dim="no_channel"),
        transforms.EnsureChannelFirstd(keys=["t1"], channel_dim="no_channel"),
        transforms.EnsureChannelFirstd(keys=["t1c"], channel_dim="no_channel"),
        transforms.EnsureChannelFirstd(keys=["t2"], channel_dim="no_channel"),
    ]
)
val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="validation",  # validation
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=8,
    download=True,  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    transform=val_transforms,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

autoencoderkl = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 128, 128),
    latent_channels=3,
    num_res_blocks=2,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)

checkpoint = torch.load('/data/users/bol/monai_playground/checkpoint/autoencoderkl_checkpoint_199.pth')
autoencoderkl.load_state_dict(checkpoint['model_state_dict'])
autoencoderkl = autoencoderkl.to(device)

with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data["flair"].to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)

unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=12,
    out_channels=6,
    num_res_blocks=2,
    num_channels=(128, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 64),
)

checkpoint_unet = torch.load('/data/users/bol/monai_playground/checkpoint/unet_checkpoint_689.pth')
unet.load_state_dict(checkpoint_unet['model_state_dict'])

unet = unet.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

low_res_scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

max_noise_level = 350

optimizer = torch.optim.Adam(unet.parameters(), lr=5e-5)

scaler_diffusion = GradScaler()

n_epochs = 1000
val_interval = 2
epoch_loss_list = []
val_epoch_loss_list = []

for epoch in range(n_epochs):
    unet.train()
    autoencoderkl.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        flair_image = batch["flair"].to(device)
        t1_image = batch["t1"].to(device)
        t1c_image = batch["t1c"].to(device)
        t2_image = batch["t2"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            with torch.no_grad():
                latent_flair = autoencoderkl.encode_stage_2_inputs(flair_image) * scale_factor
                latent_t1c = autoencoderkl.encode_stage_2_inputs(t1c_image) * scale_factor
                latent_t1 = autoencoderkl.encode_stage_2_inputs(t1_image) * scale_factor
                latent_t2 = autoencoderkl.encode_stage_2_inputs(t2_image) * scale_factor
            latent_src = torch.cat([latent_flair, latent_t1c], dim=1)
            latent_tar = torch.cat([latent_t1, latent_t2], dim=1)

            # Noise augmentation
            noise_src = torch.randn_like(latent_src).to(device)
            noise_tar = torch.randn_like(latent_tar).to(device)

            timesteps_tar = torch.randint(0, scheduler.num_train_timesteps, (latent_tar.shape[0],), device=latent_tar.device).long()
            # timesteps_t1c = torch.randint(0, scheduler.num_train_timesteps, (latent_t1c.shape[0],), device=latent_t1c.device).long()

            timesteps_src = torch.randint(0, max_noise_level, (latent_src.shape[0],), device=latent_src.device).long()
            # timesteps_t2 = torch.randint(0, max_noise_level, (t2_image.shape[0],), device=t2_image.device).long()#maybe merge two timestep

            noisy_latent_src = scheduler.add_noise(original_samples=latent_src, noise=noise_src, timesteps=timesteps_src)
            noisy_latent_tar = scheduler.add_noise(original_samples=latent_tar, noise=noise_tar, timesteps=timesteps_tar)

            # noisy_low_res_image = scheduler.add_noise(original_samples=low_res_image, noise=low_res_noise, timesteps=low_res_timesteps)

            latent_model_input = torch.cat([noisy_latent_src, noisy_latent_tar], dim=1)

            noise_pred = unet(x=latent_model_input, timesteps=timesteps_tar, class_labels=timesteps_src)
            loss = F.mse_loss(noise_pred.float(), noise_tar.float())

        scaler_diffusion.scale(loss).backward()
        scaler_diffusion.step(optimizer)
        scaler_diffusion.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        unet.eval()
        val_loss = 0
        for val_step, batch in enumerate(val_loader, start=1):
            flair_image = batch["flair"].to(device)
            t1_image = batch["t1"].to(device)
            t1c_image = batch["t1c"].to(device)
            t2_image = batch["t2"].to(device)

            with torch.no_grad():
                with autocast(enabled=True):
                    latent_flair = autoencoderkl.encode_stage_2_inputs(flair_image) * scale_factor
                    latent_t1c = autoencoderkl.encode_stage_2_inputs(t1c_image) * scale_factor
                    latent_t1 = autoencoderkl.encode_stage_2_inputs(t1_image) * scale_factor
                    latent_t2 = autoencoderkl.encode_stage_2_inputs(t2_image) * scale_factor
                    latent_src = torch.cat([latent_flair, latent_t1c], dim=1)
                    latent_tar = torch.cat([latent_t1, latent_t2], dim=1)

                    # Noise augmentation
                    noise_src = torch.randn_like(latent_src).to(device)
                    noise_tar = torch.randn_like(latent_tar).to(device)

                    timesteps_tar = torch.randint(0, scheduler.num_train_timesteps, (latent_tar.shape[0],), device=latent_tar.device).long()
                    
                    # timesteps_t1c = torch.randint(0, scheduler.num_train_timesteps, (latent_t1c.shape[0],), device=latent_t1c.device).long()

                    timesteps_src = torch.randint(0, max_noise_level, (latent_src.shape[0],), device=latent_src.device).long()
                    # timesteps_t2 = torch.randint(0, max_noise_level, (t2_image.shape[0],), device=t2_image.device).long()#maybe merge two timestep

                    noisy_latent_src = scheduler.add_noise(original_samples=latent_src, noise=noise_src, timesteps=timesteps_src)
                    noisy_latent_tar = scheduler.add_noise(original_samples=latent_tar, noise=noise_tar, timesteps=timesteps_tar)

                    # noisy_low_res_image = scheduler.add_noise(original_samples=low_res_image, noise=low_res_noise, timesteps=low_res_timesteps)

                    latent_model_input = torch.cat([noisy_latent_src, noisy_latent_tar], dim=1)

                    noise_pred = unet(x=latent_model_input, timesteps=timesteps_tar, class_labels=timesteps_src)
                    loss = F.mse_loss(noise_pred.float(), noise_tar.float())
                    val_loss += loss.item()
        val_loss /= val_step
        val_epoch_loss_list.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        # Sampling image during training
        sampling_flair_image = flair_image[0].unsqueeze(0)
        sampling_t1c_image = t1c_image[0].unsqueeze(0)
        sampling_latent_flair = autoencoderkl.encode_stage_2_inputs(sampling_flair_image) * scale_factor
        sampling_latent_t1c = autoencoderkl.encode_stage_2_inputs(sampling_t1c_image) * scale_factor
        sampling_latent_src = torch.cat([sampling_latent_flair, sampling_latent_t1c], dim=1)

        noise_src = torch.randn_like(sampling_latent_src).to(device)

        # latents = torch.randn((1, 3, 16, 16)).to(device)
        latents = torch.randn_like(sampling_latent_src).to(device)
        
        noise_level = 40
        noise_level = torch.Tensor((noise_level,)).long().to(device)
        
        noisy_latents_src = scheduler.add_noise(
            original_samples=sampling_latent_src,
            noise=noise_src,
            timesteps=torch.Tensor((noise_level,)).long().to(device),
        )

        scheduler.set_timesteps(num_inference_steps=1000)
        for t in tqdm(scheduler.timesteps, ncols=110):
            with torch.no_grad():
                with autocast(enabled=True):
                    latent_model_input = torch.cat([noisy_latents_src, latents], dim=1)
                    noise_pred = unet(
                        x=latent_model_input, timesteps=torch.Tensor((t,)).to(device), class_labels=noise_level
                    )
                latents, _ = scheduler.step(noise_pred, t, latents)

        with torch.no_grad():
            print(latents.shape)
            t1_latents = latents[:,:3,...]
            decoded = autoencoderkl.decode_stage_2_outputs(t1_latents / scale_factor)
            t2_latents = latents[:,3:,...]
            decoded_t2 = autoencoderkl.decode_stage_2_outputs(t2_latents / scale_factor)

        checkpoint = {
            'model_state_dict': unet.state_dict(),
        }

        torch.save(checkpoint, f'/data/users/bol/monai_playground/checkpoint/unet_checkpoint_{epoch}.pth')

        numpy_gt = t1_image[0, 0,:,:,32].cpu().numpy()
        numpy_gt = (numpy_gt * 255).astype(np.uint8)
        image = Image.fromarray(numpy_gt, mode='L') 
        image.save(f'/data/users/bol/monai_playground/log_image/t1_{epoch}.jpg', format='JPEG')

        numpy_pre = decoded[0, 0,:,:,32].cpu().numpy()
        numpy_pre = (numpy_pre * 255).astype(np.uint8)
        image = Image.fromarray(numpy_pre, mode='L') 
        image.save(f'/data/users/bol/monai_playground/log_image/t1_{epoch}_pre.jpg', format='JPEG')

        numpy_gt = t2_image[0, 0,:,:,32].cpu().numpy()
        numpy_gt = (numpy_gt * 255).astype(np.uint8)
        image = Image.fromarray(numpy_gt, mode='L') 
        image.save(f'/data/users/bol/monai_playground/log_image/t2_{epoch}.jpg', format='JPEG')

        numpy_pre = t2_latents[0, 0,:,:,32].cpu().numpy()
        numpy_pre = (numpy_pre * 255).astype(np.uint8)
        image = Image.fromarray(numpy_pre, mode='L') 
        image.save(f'/data/users/bol/monai_playground/log_image/t2_{epoch}_pre.jpg', format='JPEG')

        # low_res_bicubic = nn.functional.interpolate(sampling_image, (64, 64), mode="bicubic")
        # plt.figure(figsize=(2, 2))
        # plt.style.use("default")
        # plt.imshow(
        #     torch.cat([t1_image[0, 0,:,:,32].cpu(), decoded[0, 0,:,:,32].cpu()], dim=1),
        #     vmin=0,
        #     vmax=1,
        #     cmap="gray",
        # )
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()