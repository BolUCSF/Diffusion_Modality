import os, json, wandb, torch
import numpy as np
import torch.nn.functional as F
from monai import transforms, data
from monai.data import DataLoader, DistributedSampler
from monai.utils import set_determinism
from tqdm import tqdm
import torch.distributed as dist
import torch
from torch import nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from peft import LoraConfig, inject_adapter_in_model
from flow_match import FlowMatchScheduler
import matplotlib.pyplot as plt
import torch, torch.distributed as dist
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy, enable_wrap, wrap)
os.environ['MASTER_ADDR'] = 'localhost'

def get_model():
    # ▶ 1. 先把预训练权重加载成 BF16
    transformer = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,          # ← 这里已是 BF16
        cache_dir="/working/cache/huggingface/hub",
    )

    # ▶ 2. 替换 patch_embedding（保持 BF16）
    old_patch = transformer.patch_embedding
    new_patch = nn.Conv3d(
        in_channels  = old_patch.in_channels * 8,
        out_channels = old_patch.out_channels,
        kernel_size  = old_patch.kernel_size,
        stride       = old_patch.stride,
        padding      = old_patch.padding,
        bias         = old_patch.bias is not None,
    ).to(torch.bfloat16)                     # ★ 关键：立刻转 BF16
    transformer.patch_embedding = new_patch

    # ▶ 3. 替换 proj_out（保持 BF16）
    old_proj = transformer.proj_out
    new_proj = nn.Linear(
        in_features  = old_proj.in_features,
        out_features = old_proj.out_features * 4,
        bias         = True,
    ).to(torch.bfloat16)                     # ★ 同上
    transformer.proj_out = new_proj

    # ▶ 4. 保险起见，把整个模型再 cast 一遍（包含 buffer）
    transformer = transformer.to(dtype=torch.bfloat16)

    return transformer

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

epochs = 2000
# wandb.init(
#     project="SD_fsdp_A6000",
#     config={
#     "architecture": "WAN1.3B",
#     "dataset": "brats",
#     "epochs": epochs,
#     }
# )

train_json_path = './json/train.json'
with open(train_json_path) as f:
    train_files = json.load(f)
val_json_path = './json/val.json'
with open(val_json_path) as f:
    val_files = json.load(f)
train_batchsize  = 1
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
train_ds = data.Dataset(data=train_files, transform=train_transforms)
sampler_train = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank)
train_loader = DataLoader(train_ds, batch_size=train_batchsize, shuffle=False, num_workers=1, persistent_workers=True, drop_last=True, sampler=sampler_train)

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
val_ds = data.Dataset(data=val_files, transform=train_transforms)
sampler_val = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank)
val_loader = DataLoader(val_ds, batch_size=train_batchsize, shuffle=False, num_workers=1, persistent_workers=True, drop_last=True, sampler=sampler_val)

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,  # 参数 BF16
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16)

vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16, cache_dir="/working/cache/huggingface/hub")
for p in vae.parameters():       # 完全冻结
    p.requires_grad = False
# vae = vae.to(dtype=torch.bfloat16)
# vae = vae.to(torch.cuda.current_device())
with enable_wrap(
        wrapper_cls=FSDP,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device()):
    vae = wrap(
        vae,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,   # 显存≈1/卡数
        use_orig_params=True,                               # 保留原 param 句柄
        limit_all_gathers=True,                             # 额外省显存
    )
vae.eval()
# def freeze_wrap(module, recurse, nonwrapped_numel):
#     """
#     返回 True → 这个子模块需要再递归判断
#              False → 直接把它 wrap 掉
#     这里的逻辑：只要模块里没有可训练参数，就整体 wrap。
#     """
#     return not any(p.requires_grad for p in module.parameters())
# with enable_wrap(
#         wrapper_cls=FSDP,              # ← 新版必须显式给出
#         auto_wrap_policy=freeze_wrap,  # 你的 policy
#         mixed_precision=mp_policy,     # 其余 kwargs 保持
#         device_id=torch.cuda.current_device()):
#     vae = wrap(
#         vae,
#         use_orig_params=True,
#         sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
#     )

dit = get_model()
policy_blk = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={WanTransformerBlock}
)

# 3️⃣ 再写一个“黑名单”代理策略：遇到 TimeEmbedder 就直接不上 FSDP
def fsdp_policy(module, recurse, nonwrapped_numel):
    if isinstance(module, TimestepEmbedding):
        return False                          # ❌ 强行排除
    # 其余模块走正常判定
    return policy_blk(
        module=module,
        recurse=recurse,
        nonwrapped_numel=nonwrapped_numel
    )

# 4️⃣ 启用 wrap
with enable_wrap(
        wrapper_cls=FSDP,
        auto_wrap_policy=fsdp_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device()):
    dit = wrap(
        dit,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        backward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=True          # 给 StopIteration 兜底也无妨
    )


optim = torch.optim.AdamW(
        (p for p in dit.parameters() if p.requires_grad),
        lr=1e-5, betas=(0.9,0.95), weight_decay=0.01)

scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
scheduler.set_timesteps(1000, training=True)

for epoch in range(epochs):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
    for _, batch in progress_bar:
        image = batch['image']
        image = image.permute(0, 1, 4, 2, 3)
        B, C, D, H, W  = image.shape
        image = image.view(B*C, D, H, W)
        image = image.unsqueeze(1)
        image = image.repeat(1, 3, 1, 1, 1)
        image = image.to(torch.cuda.current_device(), non_blocking=True)
        image = image.to(torch.bfloat16)
        with torch.no_grad():
            latents = vae.encode(image, return_dict=True).latent_dist.sample()
            # print(torch.mean(latents), torch.std(latents))
        latents = latents.view(B, 64, -1, int(H/8), int(W/8))
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, scheduler.num_train_timesteps, (1,))
        timestep = scheduler.timesteps[timestep_id].to(torch.cuda.current_device(), non_blocking=True)
        noisy_latents = scheduler.add_noise(latents, noise, timestep)
        training_target = scheduler.training_target(latents, noise, timestep)
        mask = (torch.rand(B,4) < 0.25).float().to(torch.cuda.current_device(), non_blocking=True)
        mask = mask.unsqueeze(2)
        mask = mask.repeat(1,1,16)
        mask = mask.view(B,16*4,1,1,1)
        z_c_keep = latents
        z_c_zero = torch.zeros_like(latents).to(torch.cuda.current_device(), non_blocking=True)
        z_c_in = z_c_keep * (1-mask) + z_c_zero * mask
        inp = torch.cat([noisy_latents, z_c_in], dim=1)
        encoder_hidden_states = torch.zeros([B,32,4096], dtype=torch.bfloat16).to(torch.cuda.current_device(), non_blocking=True)
        noise_pred = dit(inp, timestep, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * scheduler.training_weight(timestep)
        loss.backward()
        optim.step()
        optim.zero_grad()
        # wandb.log({"loss": loss.item()})

#NCCL_P2P_DISABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train_fsdp.py
#OMP_NUM_THREADS=16 torchrun --standalone --nproc_per_node=8 train_fsdp.py