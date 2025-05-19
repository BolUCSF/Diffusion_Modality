import os, json, wandb, torch, deepspeed
import numpy as np
import torch.nn.functional as F
from monai import data
from transform_func import train_transforms,val_transforms
from monai.data import DataLoader, DistributedSampler
from monai.utils import set_determinism
from tqdm import tqdm
import torch.distributed as dist
import torch
from torch import nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from peft import LoraConfig, inject_adapter_in_model
from flow_match import FlowMatchScheduler
import matplotlib.pyplot as plt

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '2224'
set_determinism(42)
dist.init_process_group(backend="nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
world_size = dist.get_world_size()
epochs = 2000
if rank == 0:
    wandb.init(
        project="SD_A6000",
        config={
        "architecture": "WAN1.3B",
        "dataset": "brats",
        "epochs": epochs,
        }
    )

class FrozenVAE(nn.Module):
    def __init__(self, model_id: str, device: torch.device = None, cache_dir: str = None):
        super().__init__()
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        )
        self.latents_mean = torch.tensor([self.vae.config['latents_mean']], dtype=torch.bfloat16).view(1, 16, 1, 1, 1)
        self.latents_std = torch.tensor([self.vae.config['latents_std']], dtype=torch.bfloat16).view(1, 16, 1, 1, 1)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()
        if device is None:
            device = torch.cuda.current_device()
        self.vae.to(torch.bfloat16).to(device)
        self.latents_mean = self.latents_mean.to(torch.bfloat16).to(device)
        self.latents_std = self.latents_std.to(torch.bfloat16).to(device)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape [B, C, H, W], normalized to [-1, 1]
        Returns:
            latent: Latent tensor sampled from VAE posterior, shape [B, latent_dim, H//8, W//8]
        """
        encode_out = self.vae.encode(image, return_dict=True)
        latent = encode_out.latent_dist.sample()
        return latent
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Latent tensor of shape [B, latent_dim, H//8, W//8]
        Returns:
            image: Reconstructed image tensor of shape [B, C, H, W], normalized to [-1, 1]
        """
        decode_out = self.vae.decode(latent, return_dict=True)
        image = decode_out.sample
        return image

class UnifiedFlowNet(nn.Module):
    def __init__(self, latent_dim, input_channels = 4,  drop_prob=0.9, lora = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.model = self.init_model(lora)
        self.drop_prob = drop_prob
        

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        
        lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
        for param in model.patch_embedding.parameters():
            param.requires_grad = True

        for param in model.proj_out.parameters():
            param.requires_grad = True
        return model
    
    def init_model(self, lora):
        transformer = WanTransformer3DModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir="/working/cache/huggingface/hub")
        old_patch_embed = transformer.patch_embedding
        new_patch_embed = nn.Conv3d(
            in_channels=old_patch_embed.in_channels*8,               # 修改为新输入通道
            out_channels=old_patch_embed.out_channels,
            kernel_size=old_patch_embed.kernel_size,
            stride=old_patch_embed.stride,
            padding=old_patch_embed.padding
        )
        transformer.patch_embedding = new_patch_embed
        old_proj_out = transformer.proj_out
        new_proj_out = nn.Linear(
            in_features=old_proj_out.in_features,
            out_features=old_proj_out.out_features*4,            # 修改为新输出通道
            bias=True
        )
        transformer.proj_out = new_proj_out
        if lora:
            transformer = self.add_lora_to_model(transformer, lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out.0,linear_1,linear_2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None)
        return transformer
    
    def forward(self, z_t, timestep, encoder_hidden_states=None, z_c=None, train=True):
        if train:
            B, C, D, H, W = z_t.shape
            if encoder_hidden_states is None:
                encoder_hidden_states = torch.zeros([B,256,4096], device=z_t.device, dtype=z_t.dtype)
            if z_c is None:
                z_c_in = torch.zeros_like(z_t, device=z_t.device, dtype=z_t.dtype)
            else:
                # 以 drop_prob 随机丢弃
                mask = (torch.rand(B,self.input_channels, device=z_t.device, dtype=z_t.dtype) < self.drop_prob).float()
                mask = mask.unsqueeze(2)
                mask = mask.repeat(1,1,self.latent_dim)
                mask = mask.view(B,self.latent_dim*self.input_channels,1,1,1)
                z_c_keep = z_c
                z_c_zero = torch.zeros_like(z_c, device=z_t.device, dtype=z_t.dtype)
                z_c_in = z_c_keep * (1-mask) + z_c_zero * mask
                z_c_in = z_c_in.to(dtype=z_t.dtype)

            # 拼接输入
            inp = torch.cat([z_t, z_c_in], dim=1)
            v_pred = self.model(inp, timestep, encoder_hidden_states)
        else:
            v_pred = self.model(z_t, timestep, encoder_hidden_states)
        return v_pred.sample

def get_dataloaders(train_json_path, val_json_path, train_batchsize, world_size, rank):
    
    with open(train_json_path) as f:
        train_files = json.load(f)
    
    with open(val_json_path) as f:
        val_files = json.load(f)
    
    train_ds = data.Dataset(data=train_files, transform=train_transforms)
    sampler_train = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=train_batchsize, shuffle=False, num_workers=1, persistent_workers=True, drop_last=True, sampler=sampler_train)

    val_ds = data.Dataset(data=val_files, transform=val_transforms)
    sampler_val = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_ds, batch_size=train_batchsize, shuffle=False, num_workers=1, persistent_workers=True, drop_last=True, sampler=sampler_val)
    return train_loader, val_loader


if __name__ == "__main__":
    train_json_path = './json/train_cerebro.json'
    val_json_path = './json/val.json'
    train_batchsize  = 1
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    train_loader, val_loader = get_dataloaders(train_json_path, val_json_path, train_batchsize, world_size, rank)

    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)

    val_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    val_scheduler.set_timesteps(50, denoising_strength=1.0, shift=5.0)

    model = UnifiedFlowNet(latent_dim=16, input_channels=4, drop_prob=0.9, lora=False)

    optimizer_g = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g, T_0=10000, T_mult=2, eta_min=1e-6)

    model_engine, model_optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer_g,
        lr_scheduler=lr_scheduler,
        config="deepspeed_json/deepspeed_3.json",
    )
    model_engine.load_checkpoint("./deepspeed_checkpoint/SD_latest", tag="latest_step")
    print("model_engine loaded")
    vae = FrozenVAE(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", cache_dir="/working/cache/huggingface/hub")

    device = model_engine.device

    model_engine.train()
    train_step = 0
    val_step = 0
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        for _, batch in progress_bar:
            image = batch['image']
            path = batch['path']
            image = image.permute(0, 1, 4, 2, 3)
            B, C, D, H, W  = image.shape
            image = image.view(B*C, D, H, W)  
            image = image.unsqueeze(1).repeat(1, 3, 1, 1, 1) 
            image = image.to(torch.bfloat16).to(device)
            with torch.no_grad():
                latent = vae.encode(image)
                latent = (latent-vae.latents_mean.repeat(latent.shape[0],1,1,1,1))/vae.latents_std.repeat(latent.shape[0],1,1,1,1)
            model_optimizer.zero_grad()
            latents = latent.view(B, 64, -1, int(H/8), int(W/8))
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, scheduler.num_train_timesteps, (1,))
            timestep = scheduler.timesteps[timestep_id].to(dtype=latents.dtype, device=device)
            noisy_latents = scheduler.add_noise(latents, noise, timestep)
            training_target = scheduler.training_target(latents, noise, timestep)

            noisy_latents = noisy_latents.to(dtype = torch.bfloat16, device=device)
            timestep = timestep.to(dtype = torch.bfloat16, device=device)
            latents = latents.to(dtype = torch.bfloat16, device=device)

            noise_pred = model_engine(noisy_latents, timestep, encoder_hidden_states=None, z_c=latents)
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
            loss = loss * scheduler.training_weight(timestep)
            progress_bar.set_postfix(
                        {
                            "loss": loss.item(),
                        }
                    )
            model_engine.backward(loss)
            model_engine.step()
            train_step += 1
            if rank == 0:
                wandb.log({"loss": loss.item(), "train_step": train_step})
            # break
        progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), ncols=150)
        for val_idx, batch in progress_bar_val:
            image = batch['image']
            path = batch['path']
            image = image.permute(0, 1, 4, 2, 3)  
            B, C, D, H, W  = image.shape
            image = image.view(B*C, D, H, W)
            image = image.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            image = image.to(torch.bfloat16).to(device)
            with torch.no_grad():
                latent = vae.encode(image)
                latent = (latent-vae.latents_mean.repeat(latent.shape[0],1,1,1,1))/vae.latents_std.repeat(latent.shape[0],1,1,1,1)
                latents = latent.view(B, 64, -1, int(H/8), int(W/8))
                noise = torch.randn_like(latents)
                timestep_id = torch.randint(0, scheduler.num_train_timesteps, (1,))
                timestep = scheduler.timesteps[timestep_id].to(dtype=latents.dtype, device=device)
                noisy_latents = scheduler.add_noise(latents, noise, timestep)
                training_target = scheduler.training_target(latents, noise, timestep)

                noisy_latents = noisy_latents.to(dtype = torch.bfloat16, device=device)
                timestep = timestep.to(dtype = torch.bfloat16, device=device)
                latents = latents.to(dtype = torch.bfloat16, device=device)
                noise_pred = model_engine(noisy_latents, timestep, encoder_hidden_states=None, z_c=latents)
                loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
                loss = loss * scheduler.training_weight(timestep)
            
            progress_bar_val.set_postfix(
                        {
                            "val_loss": loss.item(),
                        }
                    )
            val_step += 1
            if rank == 0:
                wandb.log({"val_loss": loss.item(), "val_step": val_step})
            
            if val_idx == 0:
                mask = torch.tensor([[1., 1., 1., 0.],])
                mask = mask.unsqueeze(2).repeat(B,1,16)
                mask = mask.view(B,16*4,1,1,1)
                mask = mask.to(torch.bfloat16).to(device)
                with torch.no_grad():
                    latent = vae.encode(image)
                    latent = (latent-vae.latents_mean.repeat(latent.shape[0],1,1,1,1))/vae.latents_std.repeat(latent.shape[0],1,1,1,1)
                    latents_cond = latent.view(B, 64, -1, int(H/8), int(W/8))
                    z_c = latents_cond
                    z_c_keep = z_c
                    z_c_zero = torch.zeros_like(z_c, device=device, dtype=torch.bfloat16)
                    z_c_in = z_c_keep * (1-mask) + z_c_zero * mask
                    z_c_in = z_c_in.to(dtype=torch.bfloat16)

                    noise = torch.randn_like(latents_cond)
                    noise = noise.to(torch.bfloat16).to(device)
                    latents = noise
                    encoder_hidden_states = torch.zeros([B,256,4096], device=device, dtype=torch.bfloat16)
                    for progress_id, timestep in enumerate(tqdm(val_scheduler.timesteps)):
                        inp = torch.cat([latents, z_c_in], dim=1)
                        timestep = timestep.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
                        with torch.autocast("cuda"):
                            noise_pred = model_engine(inp, timestep, encoder_hidden_states, train=False)
                        latents = val_scheduler.step(noise_pred, val_scheduler.timesteps[progress_id], latents)
                    latents_out = latents.view(-1, 16, 36, 24, 24)
                    latents_out = (latents_out * vae.latents_std.repeat(latents_out.shape[0],1,1,1,1)) + vae.latents_mean.repeat(latents_out.shape[0],1,1,1,1)
                    with torch.autocast("cuda"):
                        decoded = vae.decode(latents_out)
                    output = decoded.cpu().numpy()
                    if rank == 0:
                        fig, ax = plt.subplots(1, 4)
                        ax[0].imshow(
                            batch['image'][0, 0, :, :, 70], cmap="gray"
                        )
                        ax[1].imshow(
                            batch['image'][0, 1, :, :, 70], cmap="gray"
                        )
                        ax[2].imshow(
                            output[0, 0, 70, :, :], cmap="gray"
                        )
                        ax[3].imshow(
                            output[1, 0, 70, :, :], cmap="gray"
                        )
                        wandb.log({"recon": fig, "epoch": epoch})

        save_dir = f"./deepspeed_checkpoint/SD_latest"
        model_engine.save_checkpoint(save_dir, tag="latest_step")
    dist.destroy_process_group()
    wandb.finish()
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-31 deepspeed train.py