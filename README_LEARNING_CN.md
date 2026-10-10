# DeepEP 设计与优化学习指南

本文面向希望从 CUDA、分布式通信和 MoE 系统三个层面理解 DeepEP 的读者。目标是沿着一次 token 的完整生命周期，理解每个模块为何存在，以及性能优化付出的代价。

**源码基准**：本地 `deep_ep.__version__ == '2.5.0'`，编写时提交为 `93eb6eb238127e96c6d7a4a625a6dad158348509`。本文依据当前目录的文档和源码，未实测 GPU 性能。示意图是理解模型，具体执行顺序应以对应内核为准。

本版本采用 **NCCL Gin + DeepJIT + 统一 EPBuffer API**。旧版 `Buffer`、NVSHMEM、独立 normal/low-latency API 的资料不能直接套用。安装与完整接口说明见 [README_CN.md](README_CN.md)。

## 目录

1. [先理解问题](#1-先理解问题)
2. [整体架构](#2-整体架构)
3. [拓扑与通信路径](#3-拓扑与通信路径)
4. [数据布局与缓冲区](#4-数据布局与缓冲区)
5. [Dispatch 完整流程](#5-dispatch-完整流程)
6. [Combine 与反向传播](#6-combine-与反向传播)
7. [GPU 内核优化](#7-gpu-内核优化)
8. [同步、异步与生命周期](#8-同步异步与生命周期)
9. [SM 和 QP 资源模型](#9-sm-和-qp-资源模型)
10. [其他并行与负载均衡模块](#10-其他并行与负载均衡模块)
11. [性能分析和实验](#11-性能分析和实验)
12. [分阶段阅读路线](#12-分阶段阅读路线)
13. [常见误区与自测](#13-常见误区与自测)

## 1. 先理解问题

### 1.1 MoE 的通信从哪里来

设输入 `x` 为 `[T, H]`：`T` 是本 rank 的 token 数，`H` 是 hidden dimension。Router 为每个 token 选择 `K` 个专家：

- `topk_idx`：`[T, K]`，存放全局专家编号，`-1` 表示无效选择。
- `topk_weights`：`[T, K]`，FP32 路由权重。
- `E` 个专家分布在 `P` 个 rank 上，常规布局下每个 rank 拥有 `E/P` 个专家。

```text
Router → dispatch → 本地专家 grouped GEMM → 应用路由权重 → combine
           跨 GPU           计算                          跨 GPU
```

在常规专家布局下，`owner_rank = expert_id // num_local_experts`。一个 token 的多个专家可能在同一 rank，也可能在不同 rank。

例如 `P=4, E=8, K=2`，每个 rank 两个专家：

| token 的专家选择 | 专家所在 rank | 通信层需要的目标 rank | 展开后的专家行数 |
| ---------------- | ------------- | --------------------- | ---------------- |
| `[2, 3]`       | `[1, 1]`    | `{1}`               | 2                |
| `[2, 6]`       | `[1, 3]`    | `{1, 3}`            | 2                |

**按目标 rank 去重发送，与按专家展开是两件事。** 同一 token 在一个目标 rank 上可服务多个专家；不需要在每个阶段都传输 K 份 hidden 数据。

### 1.2 DeepEP 的职责边界

DeepEP 负责 token 搬运、路由元数据传递、布局整理、结果归并与通信资源管理。它不负责 Router 算法、专家 GEMM、整个模型的自动微分封装，也不会自行决定动态专家复制计划。

DeepEP 与 DeepGEMM 的接口边界非常值得研究：通信结束后生成什么布局，才能减少 GEMM 前的额外整理？GEMM 的对齐要求，又会增加多少 padding 和内存流量？

## 2. 整体架构

```text
模型框架 / 测试调用者
    │
    ▼
deep_ep/buffers/*.py                 Python API、参数校验、handle、事件包装
    │ deep_ep._C
    ▼
csrc/python_api.cpp                 pybind 绑定入口
    │
    ├── csrc/buffers/*.hpp           缓冲区及操作编排
    ├── csrc/kernels/comm/           communicator、注册内存、通信上下文
    └── csrc/kernels/ep/*.hpp        launch 配置、模板特化、JIT 调用
                │ DeepJIT
                ▼
deep_ep/include/deep_ep/impls/       实际 CUDA 内核
    ├── layout/                     token、workspace、bucket 布局
    ├── comm/                       Gin 与 NVLink 访问封装
    └── common/ptx.cuh              TMA、barrier、原子与内存序工具
```

关键入口：

| 模块        | 阅读入口                                                                          | 要回答的问题                                 |
| ----------- | --------------------------------------------------------------------------------- | -------------------------------------------- |
| 导入与版本  | [deep_ep/__init__.py](deep_ep/__init__.py)                                   | NCCL 检查与 JIT 何时初始化？                 |
| EP API      | [deep_ep/buffers/ep.py](deep_ep/buffers/ep.py)                                     | 一次 dispatch 的输入、输出和 handle 是什么？ |
| C++ 编排    | [csrc/buffers/ep.hpp](csrc/buffers/ep.hpp)                                         | 何时分配输出、等待计数、启动尾声？           |
| JIT 启动    | [csrc/kernels/ep/dispatch.hpp](csrc/kernels/ep/dispatch.hpp)                       | 如何选择内核和模板参数？                     |
| 设备通信    | [deep_ep/include/deep_ep/comm/handle.cuh](deep_ep/include/deep_ep/comm/handle.cuh) | 同一操作如何走本地、NVLink 或 RDMA？         |
| JIT runtime | [csrc/runtime/jit.hpp](csrc/runtime/jit.hpp)                                       | 编译环境和缓存如何配置？                     |

安装时构建主机 C++ 扩展，运行时编译通信 GPU 内核。模板参数包括拓扑、hidden 字节数、top-k、容量、SM/QP 数等。特化可以消除运行时分支、固定布局并展开循环，但首次编译和多配置缓存也有成本。性能测量必须区分首次调用与缓存命中后的稳定运行。

## 3. 拓扑与通信路径

### 3.1 物理域与逻辑域

- **物理 NVLink/RDMA 域**：硬件可用的连接和通信范围。
- **逻辑 scale-up/scale-out 域**：当前算法组织通信的方式。
- **rank**：process group 内的编号；不能未经验证就当作全局 rank 或设备编号。

在 hybrid 内核中，逻辑编号为：

```text
rank_idx = scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx
```

阅读 `get_physical_domain_size()`、`get_logical_domain_size()` 和 [通信上下文](csrc/kernels/comm/context.cpp)，理解它们如何影响模式选择。逻辑 scale-up 不应在所有配置中直接等同于“一台物理机器”。

'get_physical_domain_size' 返回的是个二维结构代表(rdma_rank_num, nvlink_rank_num)， 比如两机16卡，单机走NvLink，跨机走RDMA，则为(2, 16)

'get_logical_domain_size' 如果走direct就是一维的，如果是Hybrid的话就也是二维的。

### 3.2 Direct 路径

Direct 模式按目标 rank 组织传输。设备通信封装优先使用可访问的对称指针；目标不在 NVLink 可达范围时使用 Gin RDMA。对应核心实现为 [dispatch.cuh](deep_ep/include/deep_ep/impls/ep/dispatch.cuh) 与 [combine.cuh](deep_ep/include/deep_ep/impls/ep/combine.cuh)。

优势是路径直接、分层转发管理较少；代价是一个 token 涉及远端多个 GPU 时，跨域 hidden 数据可能重复传输。

### 3.3 Hybrid 路径

```text
源 GPU → 按远端 scale-out 域组织 RDMA 传输
                     ↓
              远端转发 GPU
                     ↓ NVLink
              目标专家所在 GPU
```

Hybrid 用分层 RDMA + NVLink 通信，将远端域内的 fan-out 留给 scale-up 网络。实现见 [hybrid_dispatch.cuh](deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh)。

它更适合利用多 rail 网络，也有机会减少跨域重复 hidden 数据。但中间缓冲、转发、计数和同步增加了 HBM 流量与管理成本。是否更快取决于拓扑、token 数、路由分布和 NVLink/RDMA 带宽比例，不能只看少传了多少网络字节。

## 4. 数据布局与缓冲区

### 4.1 Token 不只是 hidden 向量

[TokenLayout](deep_ep/include/deep_ep/layout/ep/token.cuh) 将一个传输单元组织成：

```text
hidden payload | FP8 scale factors（可选）| top-k/权重/来源等 metadata | mbarrier（可选）
```

布局分别对 payload、scale 和 metadata 做对齐，再满足 RDMA 对齐要求。带 mbarrier 的布局用于相应共享内存 staging 等场景，不能把它理解为所有网络包都额外发送一个 barrier。

外部 `topk_idx` 的类型由 `deep_ep.topk_idx_t` 决定；当前传输布局内部使用 32-bit top-k 编号。阅读时要区分 API 类型和打包类型。

记录来源 token 是 combine 的基础：当前设计使用包含来源 rank/token 的元数据；参考实现以 `src_rank * num_max_tokens_per_rank + src_local_token` 构造全局 token 编号。

### 4.2 通信布局与专家布局

通信缓冲区首先服务于按 rank/channel 传输；专家 GEMM 则需要按 expert 连续排列。这是 dispatch copy epilogue 存在的重要原因。

`do_expand=True` 为每个有效 token-expert 配对生成一行，并按本地专家分组。`expert_alignment` 对每个专家的 token 数做对齐；`do_zero_padding=True` 清零专家间对齐空隙。

```text
通信侧：来源 token A、B、C，附带各自路由
专家侧：expert 0 的行 + padding | expert 1 的行 + padding | ...
```

展开提高 GEMM 可用性，但会复制 hidden 数据并产生 padding。K 较大或小 batch 时，这些额外 HBM 流量可能显著。`use_tma_aligned_col_major_sf` 则把 FP8 scale 输出整理成对应计算所需的布局。

### 4.3 容量与实际长度

`num_max_tokens_per_rank` 用于容量规划，不等于本次实际 token 数。所有 rank 必须使用一致容量。

- `do_cpu_sync=True`：获得准确接收计数，再构造相应输出；通常用于训练和 prefill。
- `do_cpu_sync=False`：按容量分配，使用 GPU 上的专家计数/前缀和确定有效区间；常用于 decoding。

“不做 CPU 同步”没有消除 GPU 间同步。“清零 padding”也不意味着容量尾部全部有效。专家 GEMM 必须遵循 handle 中的有效区间。

## 5. Dispatch 完整流程

沿着 [EPBuffer.dispatch](deep_ep/buffers/ep.py) → [C++ EP buffer](csrc/buffers/ep.hpp) → [launch_dispatch](csrc/kernels/ep/dispatch.hpp) 阅读。

以下是逻辑阶段，实际内核会并行或流水执行其中部分工作：

1. **配置**：检查输入、确定 SM/QP、解包 BF16 或 FP8+scale 输入。
2. **路由计数**：计算目标 rank 与本地专家接收量，按目标 rank 去重。
3. **槽位规划**：获得 token 在目标缓冲区中的位置，生成计数和前缀和信息。
4. **数据搬运**：hidden、scale、metadata 进入 staging/发送区，经 NVLink 或 RDMA 到达目标。
5. **远端转发**：hybrid 将到达的数据继续分发给 scale-up 域内目标 rank。
6. **完成协议**：等待计数、数据或 channel 完成信号，确保接收方能安全读取。
7. **Copy epilogue**：整理输出，必要时展开、对齐、清零 padding，并构造 `EPHandle`。

尾声实现见 [dispatch_copy_epilogue.cuh](deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.cuh)。

### 5.1 Warp 分工

Hybrid dispatch 显式设置三类 warp：

| 角色           | 工作                             |
| -------------- | -------------------------------- |
| notify warp    | 路由计数及相关通知/布局信息      |
| scale-out warp | 跨域发送和 channel 进度管理      |
| forward warp   | 处理到达数据并进行 scale-up 转发 |

源码中一个数据 warp 对应一个 channel，不同 channel 可以共享 QP。阅读时画出三个角色的生产者/消费者关系，而不是假设整个 block 每一步都同步执行相同操作。

### 5.2 EPHandle 保存什么

`EPHandle` 保存来源元数据、目标槽位、接收计数/前缀和、展开布局以及 hybrid 需要的转发元数据等。它是本次路由的数据结构，不只是完成事件。

缓存 dispatch 可重放已有布局，减少计数与 CPU 同步工作；只有路由与布局仍适用时才能复用。新的 decoding step 若重新 gating，必须重新 dispatch。原始 `topk_idx` 被 handle 引用，所有使用完成前不能修改。

## 6. Combine 与反向传播

### 6.1 Combine 的数学语义

典型 MoE 前向为：

```text
y[t] = Σ_e weight[t,e] × expert_e(x[t])
```

对于本文重点讨论的展开布局，应在专家计算路径中应用接收到的路由权重，再把加权 BF16 输出交给 combine。不要把 `combine(topk_weights=...)` 理解成自动给 hidden 输出乘 gate；该参数也用于路由权重数据的归并，例如反向中的权重梯度。

阅读入口：

- [csrc/kernels/ep/combine.hpp](csrc/kernels/ep/combine.hpp)
- [combine.cuh](deep_ep/include/deep_ep/impls/ep/combine.cuh)
- [hybrid_combine.cuh](deep_ep/include/deep_ep/impls/ep/hybrid_combine.cuh)
- [combine_reduce_epilogue.cuh](deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh)
- [combine_utils.cuh](deep_ep/include/deep_ep/impls/ep/combine_utils.cuh)

Combine 根据 handle 找到原 token，将来自不同专家的结果归并。`bias` 可在尾声加入共享专家输出或残差。

### 6.2 提前归约与精度

`allow_multiple_reduction=True` 允许多阶段归约，在分层路径上有机会先合并贡献再跨域发送，减少传输量；代价是中间精度与加法顺序可能影响结果。

设为 `False` 时，设计目标是在 combine epilogue 只进行一次归约，以改善精度，但传输数据量可能上升。应对照当前内核分支检查实际生效路径，并用测试比较误差、流量和时间。

### 6.3 为什么反向也使用这两种通信

前向 dispatch 的数据复制，在反向对应把各专家的输入梯度归并回源 token；前向 combine 的归并，在反向对应把输出梯度按原布局送回专家。

| 模型阶段         | 通信操作                |
| ---------------- | ----------------------- |
| 前向进入专家     | dispatch                |
| 前向专家输出返回 | combine                 |
| combine 的反向   | 缓存 handle 的 dispatch |
| dispatch 的反向  | combine                 |

实际自动微分封装还必须处理 gate 权重、专家计算和梯度语义。README 中的辅助函数是集成示例，不能代替完整模型 backward。

## 7. GPU 内核优化

### 7.1 TMA：降低搬运中的线程指令成本

源码通过 `tma_load_1d`、`tma_store_1d` 等封装执行异步数据搬运，使用共享内存 staging 和 mbarrier 协调完成。

重点阅读 [common/ptx.cuh](deep_ep/include/deep_ep/common/ptx.cuh)，再在 dispatch 内核中追踪：

```text
等待 staging 可复用 → 发起 load → 等待 mbarrier → 填充 metadata
    → store fence → 发起 store / commit → 等待必要完成 → 发起后续传输
```

不同路径的顺序不完全相同。学习时应逐一解释 `tma_store_wait`、`tma_store_fence` 和 phase 翻转保护的是哪段存储。

TMA 减少线程搬运指令，但仍消耗 HBM/NVLink 带宽，并占用共享内存。共享内存容量影响每个 block 可安排的 warp 数；当前 launch 配置会结合 token 大小和可用共享内存计算分工。

### 7.2 Warp 级路由去重

一个 warp 的 lane 处理 top-k 路由，通过 match、shuffle、bitmask 等操作识别相同目标并选出代表 lane，避免按每个专家重复发送完整 token。

可以先用纸笔模拟 `[2,3]` 与 `[2,6]` 两种路由，再搜索 `get_master_lane_idx`、`match`、`dst_rank_idx`，观察“专家计数”与“rank 计数”为什么不同。

### 7.3 Channel 流水与通知批处理

Hybrid 使用 channel 的接收缓冲和 signaled tail，让 forward warp 根据已到达进度处理数据。批量更新进度可以减少通知开销，但更新过慢会延迟消费，小 batch 尤其敏感。

当前 hybrid dispatch 的模板中可看到 `kScaleoutUpdateInterval` 等参数。理解其语义时同时追踪尾部不足一批的数据和 finish flag，不能只分析满批次路径。

### 7.4 通信与计算争用

EP dispatch/combine 使用 GPU SM；当前版本不提供零 SM 的 RDMA EP。更多通信 SM 可能让独立通信测试更快，也可能挤占 GEMM、HBM 和缓存，拖慢整层模型。

因此要分别衡量：

- 独立 dispatch/combine 延迟。
- 与真实专家计算重叠时的层延迟。
- 全模型吞吐或每 token 延迟。

### 7.5 FP8 的收益与成本

FP8 hidden payload 大致是 BF16 的一半字节数，但还要传 scale 与 metadata；实际流量不是严格减半。端到端还包括量化、scale 重排、专家计算和 BF16 combine。

收益更可能出现在 payload 较大且通信受带宽限制时；小 token 场景可能更受启动、计数、通知和等待影响。用真实模型的精度标准评估量化，而不只看通信测试误差。

### 7.6 确定性输出

并发接收顺序可能变化。`deterministic=True` 通过 `EPHandle.deterministic_sort` 整理相关输出及映射；展开布局按专家和来源 token 组织排序。

排序有额外成本。固定接收顺序也不自动保证整个训练的所有计算和所有加法都具有位级确定性。

## 8. 同步、异步与生命周期

### 8.1 数据就绪必须先于完成通知

CPU 线程返回、GPU kernel 完成、TMA store 完成、RDMA 数据到达和接收方可见，是不同事件。

在 [comm/handle.cuh](deep_ep/include/deep_ep/comm/handle.cuh) 中追踪 `put`、`flush`、`wait`、signal；在 PTX 工具中追踪 acquire/release 及 GPU/system scope。同步操作必须同时解决执行顺序和数据可见性。

核心不变量：**接收方观察到可消费信号时，对应数据已经可见。** 注释明确提醒不要在同一个位置混用 atomic add 和 Gin signal。理解协议后再考虑删除 fence 或减少等待。

### 8.2 Stream overlap

```text
compute stream：输入生成 ── 独立计算 ── wait ── 专家计算
                         ╲           ╱
comm stream：             dispatch ──
```

- `previous_event` 指定输入就绪依赖。
- 设置 `previous_event` 时要求 `allocate_on_comm_stream=True`。
- `async_with_compute_stream=True` 允许当前计算 stream 继续执行。
- 消费输出前调用完成事件的 wait。

没有独立计算可做时，异步 API 本身不会凭空减少延迟。

### 8.3 延迟尾声

`defer_epilogue=True` 要求异步模式，并直接返回 `EventOverlap`。第一次 `current_stream_wait()` / `wait()` 执行注册尾声并返回结果：

```python
# 示意：buffer、x、路由张量和 num_experts 由调用方准备。
event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=num_experts,
    do_expand=True, do_zero_padding=True,
    async_with_compute_stream=True, defer_epilogue=True,
)
# 在当前 compute stream 上执行不依赖本次 dispatch 结果的工作。
recv_x, recv_idx, recv_weights, handle = event.current_stream_wait()
# 用 recv_x 和 recv_weights 完成专家计算与加权，再 combine。
```

第一次 wait 的返回值应保留：Python hook 执行后被清除，后续 wait 不会重新生成同一返回值。wait 是 stream 依赖与尾声执行入口，不应一概理解为同步阻塞 CPU 直到所有 GPU 工作结束；启用准确计数的尾声可能包含 CPU 等待。

### 8.4 生命周期

每个 EP 组创建并复用缓冲区，避免每层重建 communicator 和注册内存。所有在途操作结束前不能替换或销毁缓冲区，也不能复用仍在被通信访问的存储。

阅读 [BufferBase](deep_ep/buffers/base.py)、[事件包装](deep_ep/utils/event.py) 和 C++ stream/event 实现，弄清 `record_stream` 与事件持有张量的关系。`EP_AVOID_RECORD_STREAM` 模式尤其依赖事件保持存活。Python 引用存在并不等于已经建立正确的 stream 依赖。

## 9. SM 和 QP 资源模型

### 9.1 自动 SM 估算

`EPBuffer.get_theoretical_num_sms` 根据预期路由 fan-out、NVLink/RDMA 流量和每 SM 的读写能力估算通信资源，而非每次现场 autotune。

对均匀、无重复专家选择，E 个专家划分到 G 个等大组时，K 次选择涉及的预期组数为：

```text
expected_groups = G × [1 − C(E − E/G, K) / C(E, K)]
```

源码分别估算 scale-out 域和目标 rank fan-out，再比较归一化网络耗时：

```text
rdma_traffic / rdma_bandwidth   与   nvlink_traffic / nvlink_bandwidth
```

选定瓶颈后，反推足以供应对应吞吐的 SM 读写能力，加入余量、对齐并限制到设备容量。源码同时考虑 dispatch 和 combine 的读写需求。

重要限制：模型假设 balanced gate，且源码保留对展开布局、多阶段归约和 group-limited gate 的相关 TODO。实际不均衡、短 batch、额外 HBM 竞争都可能让预测偏离最优值。

`prefer_overlap_with_compute=True` 倾向减少 SM；当前 Python 构造器默认是 True，而测试脚本参数默认是 0。比较 benchmark 与模型集成时要记录这个差异。

### 9.2 QP 为什么不是越多越好

QP（Queue Pair）是 RDMA 工作队列资源。较多 QP 可以增加并行、减少 channel 间争用，但也增加资源和 doorbell 开销。

当前 `get_theoretical_num_qps`：

- direct 倾向较少 QP，建议值为 `min(num_sms, 9)`。
- hybrid 倾向为各 channel 与通知提供独立资源，建议值为 `num_sms * 16 + 1`。
- 最终均受 `num_allocated_qps` 限制。

这些是当前实现的策略，不是普适硬件公式。调用覆盖值必须不超过初始化分配量。

## 10. 其他并行与负载均衡模块

| 模块            | 用途                                                        | 重点入口                                                                                                                                           |
| --------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| BufferAllocator | 先规划对称张量，再构造缓冲区                                | [allocator.py](deep_ep/buffers/allocator.py)                                                                                                        |
| BucketBuffer    | CP/DP 的批量 all-gather、reduce-scatter、all-reduce，实验性 | [bucket.py](deep_ep/buffers/bucket.py)                                                                                                              |
| PPBuffer        | 仅 RDMA 的流水线组内相邻 rank 收发，实验性                  | [pp.py](deep_ep/buffers/pp.py)                                                                                                                      |
| EngramBuffer    | GPU/CPU 存储上的多层远程 fetch，实验性                      | [engram.py](deep_ep/buffers/engram.py)                                                                                                              |
| EP 负载均衡原语 | 冗余专家权重预取与梯度归约                                  | [prefetch_weights.cuh](deep_ep/include/deep_ep/impls/ep/prefetch_weights.cuh)、[reduce_grads.cuh](deep_ep/include/deep_ep/impls/ep/reduce_grads.cuh) |

### 10.1 对称分配

对称内存的价值是可以用一致的相对偏移定位不同 rank 的对应对象，而不是假设不同进程的虚拟地址相同。结合 communicator 注册窗口和 `get_sym_offset/get_sym_ptr` 阅读。

BufferAllocator 让所有 rank 预先使用同样的张量形状与分配顺序，构建后得到缓冲区内 CUDA view。布局一致性是协议成立的条件。

### 10.2 Bucket 的不同优化目标

Bucket 将多个张量组织进统一存储，减少小操作管理成本。通常输入必须在 bucket 内；普通外部分配张量可用 `session()`，但 staging copy 也要计入时间。

all-gather 使用 copy engine（`num_sms=0`）；reduce-scatter 和 all-reduce 使用 SM。不要把这个零 SM 特性推断到 EP。归约操作通常接收 FP32，reduce-scatter 可选择 BF16 通信并保持 FP32 输入输出。

### 10.3 动态专家复制

当某专家过热，调用方可以将其复制到同 NVLink 域内其他 GPU 并重路由 token：

```text
负载统计 → 调用方制定复制与重路由计划 → 预取专家权重
    → 冗余专家计算 → 反向时归约冗余梯度到原专家
```

`lb_prefetch_weights` 支持权重及 scale 的匹配张量序列；`lb_reduce_grads` 把冗余 FP32 梯度累加到原始专家梯度。它们是 NVLink 域内 collective，所有 rank 都必须调用，包含没有冗余槽的 rank。

`redundancy_mapping` 的 `[r,c]` 表示域内 rank r 的冗余槽 c 对应哪个原始专家，`-1` 表示未使用。冗余存储独立于 EP dispatch/combine 区域，必须预留。梯度归约不会自动清零冗余梯度。

### 10.4 Engram 与 PP

Engram 将不规则索引读取组织为 RDMA fetch，按层提供完成钩子；学习重点是批量请求、存储位置和细粒度等待。PP 用预留收发槽管理在途 tensor；学习重点是槽位复用、生产消费节奏、大小与地址对齐。

二者的协议与 EP 路由不同，建议在理解公共通信封装后再阅读。

## 11. 性能分析和实验

### 11.1 当前学习环境

编写文档时观察到 PyTorch `2.8.0+cu129`、`torch.version.cuda == '12.9'`，找不到 `nvcc`，`torch.cuda.is_available()` 为 False，`nvidia-smi` 初始化失败。`cu129` 表示 PyTorch 构建使用的 CUDA 版本，不能据此认定已经安装对应 Toolkit。

当前仓库要求 Linux、Python 3.10+、Hopper 或更新 GPU、CUDA Toolkit 13.1+、支持 C++20 `std::format` 的编译环境、PyTorch 2.10+、NCCL 2.32.3+。因此当前可以做静态源码学习，但本文没有在此环境执行 GPU 测试。未来更换环境后应重新检查。

### 11.2 安装与测试入口

在满足环境条件后，从本目录执行；这些命令会构建/安装包或启动 GPU worker：

```bash
git submodule update --init --recursive
bash develop.sh
python tests/ep/test_ep.py --help
```

PyTorch 与 DeepEP 必须加载同一份 NCCL 动态库。安装细节、网络依赖与多节点启动约定见官方 README 和 [utils/envs.py](deep_ep/utils/envs.py)。测试脚本自己启动 worker，不应直接假设要用 torchrun 包住它。

以下为**单节点 8 GPU NVLink 环境**的示例，参数均来自当前测试脚本；其他机器须修改进程数和专家数，并满足源码中的对齐限制：

```bash
# 先做小规模正确性检查，保持完整校验。
python tests/ep/test_ep.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 2 --num-experts 32 --skip-perf-test

# 在正确性通过后采集性能和 trace。
python tests/ep/test_ep.py --num-processes 8 --num-tokens 4096 --dump-profile-traces /tmp/deepep-traces
```

正确性对照见 [utils/refs.py](deep_ep/utils/refs.py)，实际测试包含 BF16/FP8、展开与对齐、异步事件、bias 等组合。读懂 reference 比直接读复杂 kernel 更容易建立语义基准。

### 11.3 一个简单性能模型

理解稳定流水瓶颈可用以下近似下界，不能作为精确耗时公式：

```text
通信主阶段时间 ≥ max(RDMA字节 / RDMA带宽,
                    NVLink字节 / NVLink带宽,
                    HBM字节 / 有效HBM带宽)
```

启动、计数、通知、串行依赖、copy/reduce epilogue 和 CPU 等待会进一步增加时间。只有能够充分流水的部分才能用 max，而不是把全流程所有步骤都假设能重叠。

独立计算与通信理想重叠时接近 `max(T_compute, T_comm)`；争用 SM、HBM 和缓存后，二者本身都可能变慢。不要直接以两个独立 benchmark 的最大值预测真实层延迟。

### 11.4 推荐实验矩阵

每次只修改一个因素，记录硬件/拓扑、输入、完整配置、缓存是否预热以及重复测量的分布。

| 实验         | 参数/对照                              | 要验证的假设                                   |
| ------------ | -------------------------------------- | ---------------------------------------------- |
| token 数扫描 | `--num-tokens`                       | 小 batch 受固定开销限制，大 batch 转向带宽限制 |
| fan-out 扫描 | `--num-topk`                         | 流量增长由目标域去重影响，不只是线性 K 倍      |
| SM 扫描      | `--num-sms`                          | 通信延迟和真实模型重叠延迟的最优值不同         |
| QP 扫描      | `--num-qps`、`--num-allocated-qps` | 并发收益存在资源与通知代价                     |
| 拓扑模式     | `--allow-hybrid-mode 0/1`            | 分层收益是否覆盖转发与 HBM 成本；需要跨域机器  |
| 数值归约     | `--allow-multiple-reduction 0/1`     | 流量、延迟和数值误差如何变化                   |
| CPU 同步     | `--do-cpu-sync 0/1`                  | 输出容量和计数路径对小 batch 的影响            |
| 延迟尾声     | `--defer-epilogue 0/1`               | 真实独立计算能否覆盖通信/计数等待              |
| 不均衡       | `--unbalanced-ratio`                 | 解析模型与慢 rank 的偏离                       |
| 确定性       | `--deterministic`                    | 排序的延迟与额外内存成本                       |

FP8/BF16、expert alignment 的对照可先观察脚本已有枚举；若要孤立因素，可基于测试调用编写实验脚本。`--fp8-dispatch-only` 可限定 FP8 路径。不要用跳过正确性的选项掩盖改动造成的错误。

### 11.5 分析工具和读数

- **Nsight Systems / 测试 trace**：看 stream 依赖、CPU 等待、尾声位置、通信与计算重叠。
- **Nsight Compute**：看寄存器、共享内存、HBM 流量、stall 与 occupancy；分布式内核可能因 profiler replay 干扰协议，先确认工具的适用模式。
- **JIT 汇编**：用 `EP_JIT_WITH_LINEINFO=1`、`EP_JIT_DUMP_PTX=1`、`EP_JIT_DUMP_SASS=1` 辅助定位；dump 通常在缓存未命中时生成。
- **编译资源**：`EP_JIT_PTXAS_VERBOSE=1`、`EP_JIT_CHECK_NO_SPILLS=1` 辅助检查寄存器 spill。
- **拓扑与模型**：`EP_BUFFER_DEBUG=1` 查看初始化与资源估算；使用外部 NVIDIA 工具时参考 `EP_USE_NVIDIA_TOOLS=1`。

日志记录建议：

```text
提交 / GPU / NVLink / RDMA / Toolkit / PyTorch / NCCL
T,H,K,E,P / dtype / 展开与对齐 / hybrid / SM / QP
CPU同步 / 延迟尾声 / 负载分布 / 归约模式 / 是否预热
正确性误差 / dispatch / combine / epilogue / 真实层总延迟
```

带宽统计要写清分母是有效 token payload 还是含重复、metadata、padding 的实际传输量。不同口径的 GB/s 不能直接比较。

## 12. 分阶段阅读路线

### 阶段 A：建立语义（1–2 次学习）

阅读 README、本指南第 1–6 节、`utils/refs.py`、`tests/ep/test_ep.py` 的调用。

产出：手算两个 token 的目标 rank、专家展开行、来源映射和 combine 结果；解释 dispatch/combine 的返回值。

### 阶段 B：连接 Python 与 C++（2–3 次学习）

阅读 `buffers/ep.py`、`csrc/buffers/ep.hpp`、`csrc/kernels/ep/dispatch.hpp`、`combine.hpp`。

产出：画一次同步/异步 dispatch 的调用图，标出输出分配、CPU 计数等待、主内核、尾声、handle 的创建位置。

### 阶段 C：单域/direct 内核（3–4 次学习）

阅读 `layout/ep/token.cuh`、`workspace.cuh`、`impls/ep/dispatch.cuh`、`dispatch_copy_epilogue.cuh`，然后 combine。

产出：为一个 token 画出 HBM、shared memory、目标缓冲区、最终输出之间的每次搬运，标出每次同步。

### 阶段 D：Hybrid 协议（3–5 次学习）

阅读 `hybrid_dispatch.cuh`、`hybrid_combine.cuh`、`comm/handle.cuh`、`comm/barrier.cuh`。

产出：画 notify/scale-out/forward warp 的时序，解释 channel tail、finish flag、QP 与来源元数据如何配合。特别验证空 channel 和最后一个不满批次。

### 阶段 E：性能与集成（持续实验）

阅读 SM/QP 模型、事件生命周期与 profiling 工具；在合适硬件上运行实验矩阵，再接入真实专家 GEMM。

产出：用测量回答“瓶颈在哪里”“为什么这个 SM 数更好”“优化是否减少整层耗时”，避免只报告孤立通信带宽。

### 阶段 F：扩展模块

依次学习 BufferAllocator、Bucket、专家负载均衡、PP、Engram。比较它们如何复用共同的对称存储、设备通信和事件机制，以及各自的协议差异。

## 13. 常见误区与自测

### 常见误区

- 把 K 个专家理解为每个网络阶段都必须传 K 份 hidden。
- 把无 CPU 同步理解为没有通信同步，或认为容量尾部都能用于 GEMM。
- 只调用异步 API，没有独立工作或缺少输入/输出依赖。
- 认为当前 EP 的 RDMA 路径零 SM，或把 Bucket all-gather 的 copy engine 行为推广到 EP。
- 将 gate 权重直接交给 combine 并期待它自动加权 hidden 输出。
- 修改 handle 引用的路由张量，或在路由改变后继续复用旧 handle。
- 只测首轮调用，把 JIT 编译时间当成稳态通信延迟。
- 追求最大通信 SM/QP 或最高独立带宽，却没有测真实模型争用。
- 从旧版 benchmark 或实验分支推断当前主线的硬件支持与性能。

### 自测问题

1. 同一 token 选择同 rank 两个专家，为什么 rank 接收计数和 expert 接收计数不同？
2. 为什么通信层去重后，GEMM 布局仍可能复制同一 token？
3. Hybrid 到底节省哪部分网络字节，又增加哪部分 HBM 字节？
4. 接收方看到 tail 更新，凭什么确信 payload 已经可见？
5. TMA load 完成和 shared staging 可以被下一 token 覆盖，是同一个条件吗？
6. `defer_epilogue` 改变了哪部分执行时机？第一次 wait 返回什么？
7. `do_cpu_sync=False` 时，如何防止专家 GEMM 读取无效容量尾部？
8. 自动 SM 模型为什么依赖 balanced gate？不均衡会怎样破坏预测？
9. 多阶段归约如何改变流量、精度和加法顺序？
10. 反向传播为什么会交换 dispatch/combine 的角色？
11. 优化后的通信内核快了 20%，为什么模型仍可能变慢？
12. 如何用正确性 reference、trace、资源读数和真实层测量共同支持一个优化结论？

能沿源码回答这些问题，就已具备继续修改和评估 DeepEP 内核的基础。
