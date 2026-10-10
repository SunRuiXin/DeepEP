# DeepEP

DeepEP（DeepEveryParallel）是一个面向机器学习训练与推理的高性能通信库。它为 MoE 的分发（dispatch）和归并（combine）提供高吞吐、低延迟的专家并行（EP）全互联 GPU 内核，支持 FP8 分发，以及通过 NVLink 为冗余专家交换权重和梯度。此外，它还提供流水线并行（PP）、上下文并行（CP）、数据并行（DP）和远程内存访问（Engram）的实验性原语。通信内核通过 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 在运行时编译，配套扩展则在安装时构建。

## 最新动态

- **昇腾版本发布**
  - 在华为昇腾 950 NPU 上提供相同 API 和完整性能
  - 详情参见 [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)

- **V2.5 发布**
  - 将 `ElasticBuffer` 拆分为 `EPBuffer`、`EngramBuffer`、`PPBuffer` 和 `BucketBuffer`，共享 `BufferBase` 生命周期
  - 新增 `BufferAllocator`，可在构造缓冲区前规划对称张量分配
  - 通过 `BucketBuffer` 新增批量 all-gather、reduce-scatter 和 all-reduce，并支持普通 PyTorch 张量的 session
  - 为动态冗余专家新增 `EPBuffer.lb_prefetch_weights` 和 `EPBuffer.lb_reduce_grads`：专家计算前通过 NVLink 预取专家权重及量化尺度，反向传播时再将冗余专家的 FP32 梯度累加到原始专家。相关专家复制方案可参考 [MoonEP](https://github.com/MoonshotAI/MoonEP) 和 [UltraEP](https://github.com/Dots-Infra/UltraEP)，API 与集成要求见[专家负载均衡](#专家负载均衡)
  - 支持延迟执行 EP 尾声、缓存展开布局及专家间零填充
  - 支持将多层 Engram 存储在 GPU 或 CPU 上，并为每层提供一个等待钩子
  - 完全移除 V1，包括其 API、NVSHMEM 后端和旧文档；NVSHMEM 不再是依赖项

- **V2 发布**：全面重构专家并行，支持更大的 scale-up 和 scale-out 域，以及轻量级 **NCCL Gin 后端**。

### 新功能

- 通过 DeepJIT **JIT 编译通信内核**
- **NCCL Gin 后端**
  - 轻量级设备端通信 API
  - 可复用现有 NCCL communicator
- **EPv2**
  - 将高吞吐和低延迟 API 统一到单一 `EPBuffer` 接口，并提供适合分组专家 GEMM 的展开布局
  - 支持更大的 scale-up 与 scale-out 域
  - 通过解析模型计算 SM 与 QP 数量，不再需要自动调优
  - 继续支持 hybrid 和 direct 两种模式
- **Engram**（实验性，使用 RDMA）
- **PP**（实验性，使用 RDMA）
- 面向 CP 和 DP 的 **Bucket 集合通信**（实验性）：all-gather、reduce-scatter 和 all-reduce

### 注意事项

- EP dispatch 和 combine 需要使用 GPU SM；不支持零 SM 的 RDMA EP
- Bucket、Engram 和 PP 均为实验性功能
- Engram 要求 NCCL 构建提供 `ncclGinOptFlagsWarpGet`

## 快速开始

### 环境要求

- Linux
- Python 3.10 及以上
- NVIDIA Hopper 或更新架构的 GPU
- CUDA Toolkit 13.1 及以上，需支持目标 GPU，用于 DeepJIT 编译
- 支持 `std::format` 的 C++20 编译器和标准库
- 支持 CUDA 的 PyTorch 2.10 及以上
- NCCL 2.32.3 及以上
- 节点内通信需要 NVLink
- 节点间通信需要 RDMA 网络

安装过程会针对 CUDA 和 NCCL 库构建主机端 C++ 扩展。GPU 内核由 DeepJIT 在运行时针对当前设备编译，因此安装时不要求 GPU 可见，也无需设置 `TORCH_CUDA_ARCH_LIST`。运行时仍须保留 CUDA Toolkit 和主机编译器。自动带宽检测使用 `nvidia-smi` 和 `ibstat`；目前，即便通信组只使用一种传输方式，`BucketBuffer` 也要求 NVLink 与 RDMA 带宽均可被检测到。

### 安装 NCCL 依赖

安装与 CUDA 环境匹配的 NCCL 包，使 DeepEP 能找到头文件和库：

```bash
# CUDA 13.x
python -m pip install "nvidia-nccl-cu13>=2.32.3" --no-deps
# CUDA 12.x 请改用 nvidia-nccl-cu12
```

若使用自定义 NCCL，请将 `EP_NCCL_ROOT_DIR` 指向包含 `include/` 和 `lib/` 的目录。PyTorch 和 DeepEP 必须加载同一个 NCCL 动态库。使用低于 2.31 的 NCCL 头文件构建时，编译期和运行期 NCCL 版本还必须完全一致。

### 安装

```bash
# 初始化 DeepJIT 子模块
git submodule update --init --recursive

# 构建 wheel，并安装到当前 Python 环境
bash install.sh
```

随后即可在 Python 项目中导入 `deep_ep`。

### 开发与测试

```bash
# 构建扩展，并将其链接到源码树
bash develop.sh

# 运行测试
python tests/ep/test_ep.py
python tests/bucket/test_all_gather.py
python tests/bucket/test_reduce_scatter.py
python tests/bucket/test_all_reduce.py
python tests/buffer/test_allocation.py
python tests/ep/test_prefetch_weights.py
python tests/ep/test_reduce_grads.py
python tests/engram/test_engram.py
python tests/pp/test_pp.py
```

测试脚本依赖 NumPy，并会启动本地 GPU worker。多节点测试需在每个节点运行同一脚本，共享 `MASTER_ADDR` 和 `MASTER_PORT`，并将 `WORLD_SIZE` 设为节点数、`RANK` 设为节点编号。这些约定来自 [deep_ep/utils/envs.py](deep_ep/utils/envs.py) 中的 `init_dist`；必要时请适配自己的集群。除非已经设置，测试默认使用 `NCCL_IB_SL=1` 和 `EP_OVERRIDE_RDMA_SL=1`。权重预取与梯度归约测试使用单个 NVLink 域；PP 测试要求仅含 RDMA 的通信组。

## 接口与示例

### 缓冲区初始化

高吞吐和低延迟 EP 操作共用一个 `EPBuffer` 接口。直接用 MoE 配置初始化缓冲区；SM 和 QP 数量会通过解析模型估算，也可在每次调用时覆盖。

每个 EP 组应创建并复用一个缓冲区。所有 rank 的 `num_max_tokens_per_rank` 必须一致；应选择一个能覆盖预期训练、prefill 和 decoding batch 的统一容量。替换缓冲区前，必须等待所有未完成操作结束。

```python
import torch.distributed as dist
from typing import Optional
from deep_ep import EPBuffer

_buffer: Optional[EPBuffer] = None
_num_comm_sms: int = 0

def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> EPBuffer:
    """初始化或取得用于 EP 通信的 EPBuffer。"""
    global _buffer, _num_comm_sms
    required_bytes = EPBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
        return _buffer

    _buffer = EPBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
    return _buffer
```

### 在模型训练与推理中使用

训练、推理 prefill 和 decoding 使用相同的 `EPBuffer` dispatch/combine API。下面示例为分组专家 GEMM 使用展开布局，并将前向尾声延迟到框架等待结果时执行。`expert_alignment` 应设为分组 GEMM 的 token 对齐要求；使用 DeepGEMM 时，可调用 `deep_gemm.get_mk_alignment_for_contiguous_layout()`。

`do_cpu_sync=True` 会获取准确的输出尺寸和 CPU 端专家计数，通常用于训练和 prefill。使用 GPU 端接收计数（如 decoding）时应设为 `False`：输出会按配置容量分配，专家 GEMM 必须使用 `handle.psum_num_recv_tokens_per_expert` 判断有效专家区间。新的路由决策必须重新 dispatch；仅仅处于 decoding 阶段并不能让旧路由 handle 自动复用。

```python
import torch
from typing import Optional, Tuple, Union
from deep_ep import EPBuffer, EPHandle, EventHandle, EventOverlap

def dispatch_forward(x, topk_idx, topk_weights, num_experts,
                     num_max_tokens_per_rank, expert_alignment=1,
                     do_cpu_sync=True, previous_event=None) -> EventOverlap:
    return _buffer.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment, num_sms=_num_comm_sms,
        previous_event=previous_event, async_with_compute_stream=True,
        allocate_on_comm_stream=previous_event is not None,
        do_cpu_sync=do_cpu_sync, do_expand=True, do_zero_padding=True,
        use_tma_aligned_col_major_sf=True, defer_epilogue=True,
    )

def dispatch_backward(grad_recv_x, grad_recv_topk_weights,
                      handle: EPHandle, bias=None):
    return _buffer.combine(
        grad_recv_x, handle=handle, bias=bias,
        topk_weights=grad_recv_topk_weights, num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

def combine_forward(x, handle: EPHandle, bias=None,
                    previous_event=None) -> EventOverlap:
    return _buffer.combine(
        x, handle=handle, bias=bias, num_sms=_num_comm_sms,
        previous_event=previous_event, async_with_compute_stream=True,
        allocate_on_comm_stream=previous_event is not None,
        defer_epilogue=True,
    )

def combine_backward(grad_combined_x, handle: EPHandle):
    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x, handle=handle, num_sms=_num_comm_sms,
        async_with_compute_stream=True, do_expand=True,
        do_zero_padding=True, use_tma_aligned_col_major_sf=True,
    )
    return grad_x, event
```

为了重叠通信和计算，应先启动通信，在专家计算真正需要其结果时再等待：

```python
event = dispatch_forward(...)  # decoding 时使用 do_cpu_sync=False
# ... 执行独立计算 ...
recv_x, _, recv_topk_weights, handle = event.current_stream_wait()
# ... 执行专家 GEMM，并应用 recv_topk_weights 得到 expert_output ...
event = combine_forward(expert_output, handle)
# ... 执行独立计算 ...
combined_x, _ = event.current_stream_wait()
```

展开后的 `recv_topk_weights` 是一维数组，每个专家行对应一个值。专家计算应在 combine 之前应用这些路由权重；combine 会对传入的专家输出求和。combine 输入必须为 BF16。`bias` 可在 combine 尾声中加入共享专家输出或残差。

训练时需要保存前向 `handle`，供 `combine_backward` 和 `dispatch_backward` 使用。上述反向辅助函数会返回张量和 event，使用张量前必须等待 event。缓存 dispatch 可重放已保存的展开布局，避免再次执行接收计数的 CPU 同步。在 handle 的所有用途完成之前，不得修改原始 `topk_idx`。

### 延迟执行尾声

dispatch 和 combine 可同时设置 `defer_epilogue=True` 与 `async_with_compute_stream=True`。此时调用直接返回 `EventOverlap`。调用 `.wait()` 会在当前 stream 上执行延迟尾声；dispatch 返回 `(recv_x, recv_topk_idx, recv_topk_weights, handle)`，combine 返回 `(combined_x, combined_topk_weights)`。

结果由第一次 wait 产生，应保留供后续计算使用。若没有可重叠的工作，应立即调用 `.wait()`。`do_zero_padding=True` 会清除专家之间的对齐空隙；在不进行 CPU 同步时，它不会让输出容量中未使用的部分变成有效数据。

若独立计算在通信调用前已经入队，应先用 `_buffer.capture()` 捕获输入就绪事件，再将其作为 `previous_event` 传入。此时 EP 要求设置 `allocate_on_comm_stream=True`。延迟尾声仍会在 `.wait()` 时运行于当前 stream，因此中间计算产生的 combine `bias` 也能在这里被消费。

### Bucket 集合通信（实验性）

`BucketBuffer` 为 CP 和 DP 提供批量 `all_gather`、`reduce_scatter` 和 `all_reduce`。通常输入必须位于 bucket 自身的存储中。创建缓冲区前用 `BufferAllocator` 规划张量；构建后，规划中的 meta tensor 会成为缓冲区内的 CUDA view。

```python
import torch
from deep_ep import BufferAllocator, BucketBuffer

allocation_plan = BufferAllocator()
x = allocation_plan.allocate((group.size(), 1024), torch.float32)
buffer = BucketBuffer(group, allocation_plan)
x[group.rank()].fill_(group.rank())
gathered_x = buffer.all_gather(x[group.rank()]).wait()
x.fill_(1)
shard = buffer.reduce_scatter(x).wait()
x.fill_(1)
reduced_x = buffer.all_reduce(x).wait()
```

所有 rank 必须使用相同的分配计划。每个集合操作也可接收张量列表，`.wait()` 会返回相应的张量或列表。结果是缓冲区的 view；all-gather 返回展平 view。reduce-scatter 和 all-reduce 接收 FP32 输入，默认对各方贡献求和；可用 `scale` 缩放结果。reduce-scatter 还支持 `comm_precision="bf16"`，以较低精度通信，同时保持 FP32 输入输出。

对于 bucket 外部分配的张量，应在 `buffer.session()` 中调用集合操作。session 管理临时 bucket 存储及必要的 staging copy：

```python
with buffer.session():
    reduced_x = buffer.all_reduce(external_x).wait()
    # 复用 session 存储前，消费或复制 reduced_x
```

仅 NVLink 的 all-gather 路径也支持外部源和显式指定的 bucket 内 `dsts`；一般 bucket 外输入应使用 session。

三种集合操作均支持 NVLink、RDMA 和 hybrid 域。NVLink reduction 要求 NCCL multimem 支持。all-gather 由 copy engine 驱动（`num_sms=0`）；reduce-scatter 和 all-reduce 使用 GPU SM。更多示例见 [bucket 测试](tests/bucket)和[分配测试](tests/buffer/test_allocation.py)。

### 专家负载均衡

动态专家复制通过额外 GPU 分担高负载专家的计算。[MoonEP](https://github.com/MoonshotAI/MoonEP) 探索了动态冗余专家、权重预取和梯度归约的在线规划；[UltraEP](https://github.com/Dots-Infra/UltraEP)（[论文](https://arxiv.org/abs/2606.04101)）根据 gating 后的当前负载规划复制和 token 重路由，并将专家复制限制在 NVLink 域内。这些工作推动了 `EPBuffer` 冗余专家通信原语的设计。

- `lb_prefetch_weights(redundant_expert_weights, expert_weights, redundancy_mapping)` 在专家计算前，将原始专家权重推送到指定冗余槽。它接收单个张量或相匹配的非空张量序列，因此权重和量化尺度可以一起传输，且不转换 dtype。
- `lb_reduce_grads(redundant_expert_grads, expert_grads, redundancy_mapping)` 通过 NVLink 读取冗余专家 FP32 梯度，并原地累加到原始专家梯度。应使用与前向相同的 mapping，并在归约前用本地贡献或零初始化 `expert_grads`。

调用方负责提供复制计划、将 token 重路由到选定专家实例，并管理专家 GEMM。两个交换操作在每个 NVLink 域内均为 collective：域内所有 rank 都必须调用，即使某些 rank 没有分配冗余槽。操作在通信 stream 上异步运行并返回 `EventOverlap`；消费结果前需调用 `.wait()`。`previous_event` 可指定输入就绪时间，`num_sms=0` 表示采用解析估算的 SM 数量。

使用 `lb_allocation_plan_or_num_bytes` 为冗余权重和梯度预留存储，可传入 `BufferAllocator` 计划或对齐的字节数。LB 区域独立于 EP dispatch/combine 缓冲区。冗余张量必须位于该区域；原始专家权重和梯度可使用普通 CUDA 分配。NVLink 域内所有 rank 的计划必须具有相同形状和分配顺序。

`redundancy_mapping` 必须是形状为 `[num_nvlink_ranks, num_redundant_experts]` 的连续 CUDA int32 张量，且域内所有 rank 内容一致。条目 `[r, c]` 将某专家分配到域内 rank `r` 的冗余槽 `c`；`-1` 表示未使用。专家 ID 为 `owner_rank * num_local_experts + local_expert_idx`。副本应分配给其他 rank，不能把本 rank 专家放入本 rank 的冗余槽。

所有权重和梯度张量都必须是连续 CUDA 张量。权重第一维为 `num_local_experts`，冗余张量第一维为 `num_redundant_experts`；尾部形状可不同，但每对匹配张量的单专家字节数必须相同，且是 `deep_ep.get_num_tma_alignment()`（32 字节）的倍数。所有 rank 的 `num_local_experts` 必须一致。梯度归约接收二维 FP32 张量；应将每个专家参数展平成满足相同字节对齐的一行。

```python
import torch
from deep_ep import BufferAllocator, EPBuffer

lb_plan = BufferAllocator()
redundant_weights = lb_plan.allocate((num_redundant_experts, 1024), torch.bfloat16)
redundant_grads = lb_plan.allocate((num_redundant_experts, 1024), torch.float32)
buffer = EPBuffer(group, num_max_tokens_per_rank=num_max_tokens_per_rank,
                  hidden=hidden, num_topk=num_topk,
                  lb_allocation_plan_or_num_bytes=lb_plan)
weights_ready = buffer.lb_prefetch_weights(
    redundant_weights, expert_weights, redundancy_mapping)
weights_ready.wait()
# ... 使用预取权重执行专家计算 ...
grads_ready = buffer.lb_reduce_grads(
    redundant_grads, expert_grads, redundancy_mapping)
grads_ready.wait()
# expert_grads 现已包含所有冗余实例的贡献
```

交换完成前，mapping 和张量存储必须保持有效。若跨层或 microbatch 复用冗余存储，应在反向计算前恢复所需权重，并在复用输入前完成梯度归约。归约不会清除冗余梯度；下次累加前需覆盖或清零。完整示例见[权重预取测试](tests/ep/test_prefetch_weights.py)和[梯度归约测试](tests/ep/test_reduce_grads.py)。

### Engram（实验性）

`EngramBuffer` 支持 GPU/CPU 存储以及通过 RDMA 进行多层 fetch。先用 `get_storage_size_hint` 和 `get_theoretical_config` 确定缓冲区及 QP 大小，再通过 `set_config` 和 `write` 填充各层表。

`fetch(indices)` 接收形状为 `[num_layers, num_tokens, num_entries_per_token]` 的 int32 张量，并为每层返回一个完成钩子。使用某层获取的数据前需调用对应钩子。BF16、FP8 和 CPU 存储示例见 [Engram 测试](tests/engram/test_engram.py)。

### 流水线并行（实验性）

`PPBuffer` 在仅 RDMA 的流水线组中，为相邻 rank 提供 `send(x, dst_rank_idx)` 和 `recv(x, src_rank_idx)`。构造时用 `num_max_tensor_bytes` 和 `num_max_inflight_tensors` 预留收发槽。输入输出必须是连续 CUDA 张量，指针及大小均按 32 字节对齐。用法见 [PP 测试](tests/pp/test_pp.py)。

### 环境变量

请在导入 `deep_ep` 和创建缓冲区前设置运行时变量。除特别说明外，标志使用 `0`/`1`。以下默认值假定包中没有保存构建期默认值。

**运行时与网络**

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_BUFFER_DEBUG` | 未设置 | 设为 `1` 输出初始化、拓扑、缓冲区大小和 SM 估算诊断；不设置则禁用。Python 端检查也会把字符串 `"0"` 视为启用。 |
| `EP_SUPPRESS_NCCL_CHECK` | `0` | 跳过导入期的重复 NCCL 库检查以及与所选安装的二进制一致性检查；不会跳过 C++ 设备 communicator 兼容性检查。 |
| `EP_AVOID_RECORD_STREAM` | `0` | 对 EP 和 bucket 操作，在完成事件中保留张量，而不调用 `record_stream`。通信完成前需保持事件存活。 |
| `EP_REUSE_NCCL_COMM` | `1` | 若 PyTorch process group 后端暴露 `_comm_ptr`，则复用其 NCCL communicator；否则创建 DeepEP 管理的 communicator。 |
| `EP_DEFAULT_RDMA_SL` | 未设置 | 未传缓冲区 `sl_idx` 时设置 Gin traffic class/service level；两者均未设置则使用 NCCL 默认值。 |
| `EP_OVERRIDE_RDMA_SL` | 未设置 | 覆盖 `EP_DEFAULT_RDMA_SL` 和缓冲区 `sl_idx`。 |
| `EP_DISABLE_GIN` | `0` | 仅使用 NVLink 时跳过 Gin 初始化。RDMA 操作需要 Gin；此标志不会选择其他 RDMA 后端。 |
| `EP_NUM_MAX_LOCAL_RANKS` | `16` | 仅用于 Engram：估算 hybrid 模式下用于 `NCCL_WIN_STRIDE` 的注册存储；这是容量估计，不是 rank 数量限制。 |

**JIT 编译**

DeepJIT 优先读取 `EP_JIT_*`，再将对应 `DJ_JIT_*` 作为全局回退。应在首次内核编译前配置。显式的 `EP_JIT_*`（包括打包默认值）优先于 `DJ_JIT_*`。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_JIT_DEBUG` | `0` | 启用编译命令、内核加载、PTXAS、源码行信息及 PTX/SASS dump 诊断。 |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | 缓存根目录或用冒号分隔的根目录列表；按序搜索，并将新产物写入第一个目录。 |
| `EP_JIT_NVCC_COMPILER` | 已检测 Toolkit 的 `bin/nvcc` | 覆盖 NVCC 可执行文件；仍须能发现有效 CUDA Toolkit 根目录。 |
| `EP_JIT_CPP_STANDARD` | `20` | 传给 NVCC 的 C++ 标准；DeepEP 要求 C++20 或更高。 |
| `EP_JIT_PRINT_COMPILER_COMMAND` | `0` | 输出编译器和反汇编命令。 |
| `EP_JIT_PRINT_LOAD_TIME` | `0` | 输出内核二进制加载时间。 |
| `EP_JIT_PTXAS_VERBOSE` | `0` | 启用并输出详细 PTXAS 信息。 |
| `EP_JIT_CHECK_NO_SPILLS` | `0` | 拒绝存在寄存器 spill 的编译内核。 |
| `EP_JIT_CHECK_NO_LOCAL_MEMORY` | `0` | 拒绝使用 local memory 的编译内核。 |
| `EP_JIT_WITH_LINEINFO` | `0` | 嵌入用于性能分析的源码行信息。 |
| `EP_JIT_DUMP_ASM` | `0` | 缓存未命中时同时生成 PTX 和 SASS。 |
| `EP_JIT_DUMP_PTX` | `0` | 缓存未命中时生成 PTX。 |
| `EP_JIT_DUMP_SASS` | `0` | 缓存未命中时生成 SASS；需要 Toolkit 的 `cuobjdump`。 |
| `EP_GIN_GDAKI_DEBUG` | `0` | 编译 JIT 内核时启用 NCCL Gin GDAKI 设备调试。 |

DeepJIT 依次通过 `CUDA_HOME`、`CUDA_PATH`、`PATH` 中的 `nvcc` 和 `/usr/local/cuda` 查找 CUDA Toolkit。

**构建与依赖发现**

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_NCCL_ROOT_DIR` | 自动检测 | 包含 `include/` 和 `lib/` 的 NCCL 安装目录，构建和导入时使用；优先级高于 `NCCL_DIR` 和 NVIDIA Python 包发现。 |
| `EP_NUM_TOPK_IDX_BITS` | `64` | 构建期 top-k 索引宽度（`32` 或 `64`）。路由索引应使用导出的 `deep_ep.topk_idx_t`；修改后需重建扩展。 |

包构建时设置的下列变量会保存为导入期默认值：`EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_JIT_CPP_STANDARD`、`EP_NUM_TOPK_IDX_BITS`、`EP_NCCL_ROOT_DIR`、`EP_DEFAULT_RDMA_SL` 和 `EP_OVERRIDE_RDMA_SL`。导入时已有的环境值优先。运行时覆盖 `EP_NUM_TOPK_IDX_BITS` 不会改变已编译的索引类型。

**测试性能分析**

以下标志影响 [deep_ep/utils/testing.py](deep_ep/utils/testing.py) 中的 `bench_kineto`，不影响生产通信：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `EP_USE_NVIDIA_TOOLS` | `0` | 使用 Nsight 或 Compute Sanitizer 时跳过内部 profiler；启用时内部计时为占位值。 |
| `EP_DISABLE_BARRIER_PROFILING` | `0` | 禁用每次 profile 迭代前插入的 barrier 和延迟。 |

## 网络配置

DeepEP 已在 InfiniBand 网络上充分测试，理论上也兼容融合以太网上的 RDMA（RoCE）。

### 流量隔离

InfiniBand 通过虚拟通道（VL）支持流量隔离。为避免不同流量互相干扰，建议将专家并行流量和其他流量分配到不同 VL。通过缓冲区的 `sl_idx` 选择 RDMA service level，优先级为：`EP_OVERRIDE_RDMA_SL` > `sl_idx` > `EP_DEFAULT_RDMA_SL` > NCCL 默认值。网络的 SL-to-VL 映射决定实际承载流量的虚拟通道。

### 自适应路由

自适应路由是 InfiniBand 交换机提供的高级功能，可将流量均匀分配到多条路径。尽管它会增加额外延迟，仍建议在各种网络负载条件下启用。

### 拥塞控制

对于追求最大带宽的工作负载，建议在验证网络配置后禁用拥塞控制。若拥塞不可避免，应将这些工作负载放到较低优先级的虚拟通道。DeepEP 不负责配置网络拥塞控制。

### PCI 原子模式

若硬件支持，建议使用以下命令设置网卡的 `PCI_ATOMIC_MODE`，以提升 RDMA 原子操作性能：

```bash
# 将 mlx5_0 替换为目标网卡
sudo mlxconfig -y -d mlx5_0 set PCI_ATOMIC_MODE=4
```

## 实验性分支

以下链接指向独立实现和研究分支，其功能及依赖仅适用于对应分支；与当前 API 集成前请阅读各分支说明。

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
  - 去除 PyTorch 张量与通信缓冲区之间的复制，显著降低普通内核的 SM 使用量
  - 由**腾讯网络平台部**贡献
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
  - 使用低延迟协议，消除 RDMA 原子操作带来的额外 RTT 延迟
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
  - 使用 TMA 指令的新后端，降低 SM 使用量并支持更大的 NVLink 域
  - 支持单 batch 场景下细粒度通信计算重叠
  - 支持无 NVLink 环境的 PCIe 内核
  - 支持 NVFP4 数据类型
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
  - 此优化系列由**蚂蚁集团网络平台部**贡献
  - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347)：解耦通信内核执行和网卡 token 传输，消除 RDMA 路径的 SM 使用
  - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483)：通过信号机制重叠 Down GEMM 与 Combine Send，降低端到端延迟
  - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500)：通过 rail 优化转发和数据合并优化跨节点 LL 通信
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
  - 基于 [MORI](https://github.com/ROCm/mori) 后端支持 ROCm/AMD GPU（低延迟模式）
- [nvDev](https://github.com/deepseek-ai/DeepEP/tree/nvDev)
  - 基于 V2、采用最新 CUDA 功能的分支，例如可改善小 token 规模延迟的 Compute Fabric Transport（CFT）

## 社区分支

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep)：支持异构 GPU（如 NVIDIA、AMD）和网卡（如 EFA、Broadcom、CX7）
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP)：为 IBRC 传输增加多 QP 方案和双端口网卡支持
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace)：用于高效、精确定位慢 rank 的诊断分析器
- [ROCm/mori](https://github.com/ROCm/mori)：AMD 面向高性能 AI 工作负载（如 Wide EP、KVCache 传输和集合通信）的下一代通信库

## 致谢

DeepEP 构建于 [NCCL](https://github.com/nvidia/nccl) Gin 后端之上。感谢 [@sjeaugey](https://github.com/sjeaugey)、[@pakmarkthub](https://github.com/pakmarkthub)、[@sb17v](https://github.com/sb17v)、[@xiaofanl-nvidia](https://github.com/xiaofanl-nvidia) 及 NCCL 团队的支持！

同时感谢 [MoonEP](https://github.com/MoonshotAI/MoonEP) 和 [UltraEP](https://github.com/Dots-Infra/UltraEP) 在动态专家复制及支撑专家负载均衡的权重/梯度交换方面所做的工作。

## 许可证

本代码仓库采用 [MIT 许可证](LICENSE)发布。

## 引用

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
