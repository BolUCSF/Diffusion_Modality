import os, json, wandb, torch, deepspeed
import numpy as np
import torch.nn.functional as F
from monai import transforms, data
from monai.data import DataLoader, DistributedSampler
from monai.utils import set_determinism
from tqdm import tqdm, trange
import torch.distributed as dist
import torch
from torch import nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from peft import LoraConfig, inject_adapter_in_model
from flow_match import FlowMatchScheduler
import matplotlib.pyplot as plt

def add_lora_to_model(model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
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

def init_model(lora):
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
        transformer = add_lora_to_model(transformer, lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out.0,linear_1,linear_2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None)

    return transformer

if __name__ == "__main__":
    model = init_model(lora=False)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_g, T_0=10000, T_mult=2, eta_min=1e-6)
    model_engine, model_optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer_g,
        lr_scheduler=lr_scheduler,
        config="deepspeed_json/deepspeed_3.json",
    )
    model_engine.train()
    for i in trange(100):
        model_optimizer.zero_grad()
        fake_input = torch.randn(1, 16*8, 32, 24, 24).to(torch.bfloat16).to(model_engine.device)
        fake_target = torch.randn(1, 16*4, 32, 24, 24).to(torch.bfloat16).to(model_engine.device)
        timestep = torch.randn((1,)).to(model_engine.device)
        encoder_hidden_states=torch.zeros([1,256,4096]).to(torch.bfloat16).to(model_engine.device)
        output = model_engine(fake_input, timestep, encoder_hidden_states).sample
        print(output.shape)
        loss = F.mse_loss(output, fake_target)
        model_engine.backward(loss)
        model_engine.step()

