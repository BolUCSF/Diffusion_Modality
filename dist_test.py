# dist_test.py
import os, torch, torch.distributed as dist

def main():
    # ----------------- 初始化 -----------------
    dist.init_process_group(backend="nccl")          # GPU 训练默认用 NCCL
    rank        = dist.get_rank()                    # 全局编号
    world_size  = dist.get_world_size()              # 进程总数
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ----------------- 打印基本信息 -----------------
    if rank == 0:
        print(f"\n✓ Distributed initialized — world_size={world_size}\n")

    print(f"[Rank {rank:02d}] local_rank={local_rank}  device={device}")

    # ----------------- 做一次 all_reduce -----------------
    x = torch.ones(1, device=device) * rank          # tensor=[rank]
    dist.all_reduce(x, op=dist.ReduceOp.SUM)         # 所有 rank 求和
    expected = world_size * (world_size - 1) / 2     # 0+1+…+(n-1)
    assert x.item() == expected, f"all_reduce failed: {x.item()} != {expected}"

    if rank == 0:
        print("\n✓ all_reduce result correct, basic comm OK 🎉")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    #CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 dist_test.py