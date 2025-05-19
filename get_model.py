from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

model_path = '/working/Project/StableDiffusion/Stable_diffusion/deepspeed_checkpoint/SD_latest'
convert_zero_checkpoint_to_fp32_state_dict(model_path,model_path+'/SD', tag = 'latest_step')