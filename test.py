"""最小 BF16 dispatch/combine 示例：每个 rank 一个恒等专家。

安装当前 DeepEP 并满足其 GPU/软件依赖后，在本目录运行：
    torchrun --standalone --nproc_per_node=2 test.py

示例面向单节点 NVLink GPU；进程数应与使用的 GPU 数一致。
"""

import os

import torch
import torch.distributed as dist


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请先检查 GPU、驱动和 PyTorch 环境。")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("请用 torchrun --standalone --nproc_per_node=2 test.py 启动。")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 在设好当前设备后导入；DeepEP 导入时会检查 NCCL 并初始化 JIT。
    import deep_ep

    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    buffer = None
    try:
        group = dist.group.WORLD
        rank = dist.get_rank()
        num_experts = dist.get_world_size()
        num_tokens, hidden = 128, 7168
        torch.manual_seed(42 + rank)

        buffer = deep_ep.EPBuffer(
            group,
            num_max_tokens_per_rank=num_tokens,
            hidden=hidden,
            num_topk=1,
            explicitly_destroy=True,
        )

        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        topk_idx = torch.randint(
            num_experts, (num_tokens, 1),
            device="cuda", dtype=deep_ep.topk_idx_t,
        )
        topk_weights = torch.ones(num_tokens, 1, device="cuda", dtype=torch.float32)

        # 同步 API：默认在当前计算 stream 上建立结果就绪依赖。
        recv_x, _, recv_weights, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            do_expand=True,
            expert_alignment=1,
            do_cpu_sync=True,
        )

        # 恒等专家代替 GEMM；在 combine 之前应用 gate 权重。
        expert_output = (recv_x.float() * recv_weights[:, None]).to(torch.bfloat16)
        combined_x, _, _ = buffer.combine(expert_output, handle=handle)

        # K=1、权重=1，恒等专家经分发和归并后应完全恢复输入。
        torch.testing.assert_close(combined_x, x, rtol=0, atol=0)
        torch.cuda.synchronize()
        print(
            f"rank {rank}: PASS | input={tuple(x.shape)} "
            f"received={tuple(recv_x.shape)} output={tuple(combined_x.shape)}",
            flush=True,
        )
        dist.barrier()
    finally:
        # 先结束在途 GPU 操作，再释放 DeepEP 与分布式资源。
        if buffer is not None:
            torch.cuda.synchronize()
            buffer.destroy()
        deep_ep.destroy_all_managed_nccl_comm()
        dist.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
