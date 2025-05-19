import torch
import os
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy, enable_wrap, wrap)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from diffusers import AutoencoderKLWan
import warnings
from functools import partial

# Suppress potential warnings related to FSDP and unused parameters
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.fsdp")

# Initialize distributed environment
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

# ----------------- 1. 加载模型 -----------------
# Load in float32 first, we'll handle precision conversion with FSDP
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16,  # Load in full precision first
    cache_dir="/working/cache/huggingface/hub",
)

# For inference-only mode, we don't need to wrap with FSDP at all
# But if you need to use the model in a mixed training setup, keep parameters frozen
for p in vae.parameters():
    p.requires_grad = False

# ----------------- 2. FSDP 包装（针对推理模式优化） -----------------
# Define mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
)

from diffusers.models.autoencoders.autoencoder_kl_wan import WanEncoder3d, WanDecoder3d, WanAttentionBlock
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

policy_blk = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={WanAttentionBlock}
)

def fsdp_policy(module, recurse, nonwrapped_numel):
    # 不 wrap 任何 CausalConv3d
    if not isinstance(module, WanAttentionBlock):
        return False
    # 其余交给原 policy 判断
    return policy_blk(
        module=module,
        recurse=recurse,
        nonwrapped_numel=nonwrapped_numel,
    )

with enable_wrap(
        wrapper_cls=FSDP,
        auto_wrap_policy=fsdp_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True):
    vae = wrap(
        vae,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        backward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=True          # 给 StopIteration 兜底也无妨
    )


# Set to eval mode after FSDP wrapping
vae.eval()

if local_rank == 0:
    print("VAE model wrapped with FSDP successfully.")

# ----------------- 3. 推理调用 -----------------
with torch.no_grad():
    # Create input tensor directly in the right dtype on the right device
    mri_tensor = torch.rand(
        [1, 3, 141, 192, 192],
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
    )
    
    # Add error handling for the inference
    try:
        # The encode method will handle dtype conversions internally
        latents = vae.encode(mri_tensor).latent_dist.sample()
        print(f"[rank{local_rank}]: Latents shape: {latents.shape}, dtype: {latents.dtype}")
    except Exception as e:
        print(f"[rank{local_rank}]: Error during inference: {str(e)}")
        
        # Try with float32 if bfloat16 fails
        if "CUDA error" in str(e) or "NotImplementedError" in str(e):
            print(f"[rank{local_rank}]: Retrying with float32 input...")
            mri_tensor = mri_tensor.to(torch.float32)
            latents = vae.encode(mri_tensor).latent_dist.sample()
            print(f"[rank{local_rank}]: Succeeded with float32. Latents shape: {latents.shape}")

# Synchronize all processes before cleanup
dist.barrier()
dist.destroy_process_group()