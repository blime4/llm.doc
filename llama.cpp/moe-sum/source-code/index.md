# moe_sum 算子源码级文档

> **完整实现解析 - 适合开发者、贡献者和性能工程师**
>
> 本文档详细解析 `moe_sum` 算子在 llama.cpp 中的完整实现，包括 API 定义、CPU 实现、CUDA/DLCU 内核实现、调度机制和性能优化策略。

---

## 目录

1. [概述](#概述)
2. [API 定义](#api-定义)
3. [内存布局与数据流](#内存布局与数据流)
4. [CPU 实现](#cpu-实现)
5. [CUDA/DLCU 实现](#cudadlcu-实现)
6. [调度机制](#调度机制)
7. [性能分析与优化](#性能分析与优化)
8. [完整调用链](#完整调用链)

---

## 概述

### 什么是 moe_sum

`moe_sum` 是混合专家模型（Mixture of Experts, MoE）中的关键算子，用于将多个专家的输出按位置相加，合并成单个输出。

### 数学定义

给定输入张量 `x` 形状为 `[hidden_dim, n_experts_used, n_tokens]`，输出张量 `y` 形状为 `[hidden_dim, n_tokens]`：

```
y[d, t] = Σ_{k=0}^{n_experts_used-1} x[d, k, t]
```

其中：
- `d ∈ [0, hidden_dim)` - 隐藏维度索引
- `k ∈ [0, n_experts_used)` - 使用的专家索引
- `t ∈ [0, n_tokens)` - token 索引

### 在 llama.cpp 中的位置

```
ggml/include/ggml.h          # API 枚举定义
ggml/src/ggml.c              # API 函数实现
ggml/src/ggml-cpu/ops.cpp    # CPU 实现
ggml/src/ggml-dlcu/dl-moesum.cu    # DLCU CUDA 内核实现
ggml/src/ggml-dlcu/dl-moesum.cuh   # DLCU CUDA 头文件
ggml/src/ggml-cpu/ggml-cpu.c       # CPU 调度
ggml/src/ggml-cuda/ggml-cuda.cu    # CUDA 调度
src/llama-model.cpp          # MoE 模型使用
```

---

## API 定义

### 1. 操作枚举

**文件**: `ggml/include/ggml.h:569-571`

```cpp
enum ggml_op {
    // ... 其他操作 ...
    GGML_OP_GLU,
#ifdef GGML_USE_DLCU
    GGML_OP_MOE_SUM,     // moe_sum 操作枚举
#endif
    GGML_OP_COUNT,
};
```

**说明**:
- `GGML_OP_MOE_SUM` 仅在 `GGML_USE_DLCU` 宏定义时可用
- DLCU (DeepLink Cuda) 是华为昇腾 NPU 的 CUDA 兼容层

### 2. API 函数声明

**文件**: `ggml/src/ggml.c:5869-5881`

```cpp
#ifdef GGML_USE_DLCU
struct ggml_tensor * ggml_moe_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_expert_used) {
    GGML_ASSERT(a->ne[1] == n_expert_used);
    const int64_t ne[2] = {a->ne[0], a->ne[2]};
    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, 2, ne);

    result->op     = GGML_OP_MOE_SUM;
    result->src[0] = a;

    return result;
}
#else
struct ggml_tensor * ggml_moe_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_expert_used) {
    (void)ctx;
    (void)a;
    (void)n_expert_used;
    GGML_ASSERT(false && "ggml_moe_sum requires GGML_USE_DLCU");
    return NULL;
}
#endif
```

**参数说明**:
- `ctx`: GGML 上下文，用于内存分配
- `a`: 输入张量，形状为 `[hidden_dim, n_expert_used, n_tokens]`
- `n_expert_used`: 使用的专家数量

**输出**:
- 返回形状为 `[hidden_dim, n_tokens]` 的张量

**张量结构设置**:
```cpp
result->op     = GGML_OP_MOE_SUM;    // 设置操作类型
result->src[0] = a;                   // 设置输入源
```

---

## 内存布局与数据流

### 输入张量布局

**形状**: `[hidden_dim, n_experts_used, n_tokens]`

```
内存布局 (Row-Major):
┌─────────────────────────────────────────────────────────────┐
│  输入张量 x (3D)                                             │
│                                                             │
│  ne[0] = hidden_dim (D)  - 最内层维度                       │
│  ne[1] = n_experts_used (K) - 中间层维度                    │
│  ne[2] = n_tokens (T)      - 最外层维度                     │
│                                                             │
│  nb[0] = sizeof(type)        - 元素字节大小                 │
│  nb[1] = D * nb[0]           - 专家维度步长                 │
│  nb[2] = K * D * nb[0]       - token 维度步长               │
└─────────────────────────────────────────────────────────────┘
```

### 访问模式

**C 代码访问元素**:
```cpp
// 访问 x[d, k, t]
size_t offset = t * nb[2] + k * nb[1] + d * nb[0];
element = *((char*)data + offset);
```

### 数据流示意图

```
┌─────────────────────────────────────────────────────────────────┐
│                        moe_sum 数据流                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: [D=4096, K=4, T=512]                                    │
│                                                                 │
│   Token 0                    Token 1                    Token T│
│  ┌─────────┐                ┌─────────┐                ┌───────┐│
│  │ Expert0 │                │ Expert0 │                │       ││
│  │ Expert1 │   ────────▶    │ Expert1 │   ────────▶    │       ││
│  │ Expert2 │                │ Expert2 │                │       ││
│  │ Expert3 │                │ Expert3 │                │       ││
│  └─────────┘                └─────────┘                └───────┘│
│     [D, 4]                     [D, 4]                     [D,4] │
│       │                          │                          │   │
│       ▼                          ▼                          ▼   │
│   求和: Σ                      求和: Σ                     求和: Σ│
│       │                          │                          │   │
│       ▼                          ▼                          ▼   │
│   [D, 1]                      [D, 1]                     [D,1] │
│                                                                 │
│  输出: [D=4096, T=512]                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## CPU 实现

### 实现

**文件**: `ggml/src/ggml-cpu/ops.cpp:11013-11063`

```cpp
#ifdef GGML_USE_DLCU
void ggml_compute_forward_moe_sum(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    // [hidden_dim, n_experts_used, tokens]
    ggml_tensor * src0 = dst->src[0];
    const int n_expert_used = src0->ne[1];
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // 初始化输出为 0
    memset(dst->data, 0, ggml_nbytes(dst));

    // 创建目标视图，用于累积求和
    ggml_tensor dst_view = {
        /*.type         =*/ dst->type,
        /*.buffer       =*/ dst->buffer,
        /*.ne           =*/ { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] },
        /*.nb           =*/ { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3] },
        /*.op           =*/ dst->op,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { NULL },
        /*.view_src     =*/ { NULL },
        /*.view_offs    =*/ 0,
        /*.data         =*/ dst->data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        /*.padding      =*/ { 0 },
    };

    dst_view.src[1] = &dst_view;  // 设置 add 操作的目标

    // 对每个专家，创建输入视图并累加到目标
    for (int i = 0; i < n_expert_used; i++) {
        ggml_tensor src0_view = {
            /*.type         =*/ src0->type,
            /*.buffer       =*/ src0->buffer,
            /*.ne           =*/ { src0->ne[0], src0->ne[2], src0->ne[3], 1 },
            /*.nb           =*/ { src0->nb[0], src0->nb[2], src0->nb[3], src0->nb[3] },
            /*.op           =*/ GGML_OP_NONE,
            /*.op_params    =*/ { 0 },
            /*.flags        =*/ 0,
            /*.src          =*/ { NULL },
            /*.view_src     =*/ { NULL },
            /*.view_offs    =*/ 0,
            /*.data         =*/ ((uint8_t*)src0->data) + i * src0->nb[1],
            /*.name         =*/ { 0 },
            /*.extra        =*/ NULL,
            /*.padding      =*/ { 0 },
        };
        dst_view.src[0] = &src0_view;
        ggml_compute_forward_add(params, &dst_view);
    }
}
#endif
```

### CPU 实现解析

**核心思想**: 利用现有的 `ggml_compute_forward_add` 函数，通过张量视图切片实现专家维度的累加。

**关键步骤**:

1. **初始化输出为 0**
   ```cpp
   memset(dst->data, 0, ggml_nbytes(dst));
   ```

2. **创建目标视图** (`dst_view`)
   - 用于 `add` 操作的目标累积区
   - `dst_view.src[1] = &dst_view` 设置为 in-place 加法

3. **循环处理每个专家**
   - 创建输入视图 `src0_view`，指向当前专家的数据
   - `data = ((uint8_t*)src0->data) + i * src0->nb[1]` - 偏移到第 i 个专家
   - 调用 `ggml_compute_forward_add(params, &dst_view)` 累加

**视图技巧**:
```cpp
// 原始输入: [D, K, T]
// 输入视图: [D, T] - 选择第 i 个专家
/* .ne = */ { src0->ne[0], src0->ne[2], ... }  // D, T
/* .nb = */ { src0->nb[0], src0->nb[2], ... }  // 跳过专家维度
```

---

## CUDA/DLCU 实现

### 文件结构

```
ggml/src/ggml-dlcu/
├── dl-moesum.cuh    # 头文件，声明函数
└── dl-moesum.cu     # 实现，包含多个内核
```

### 头文件

**文件**: `ggml/src/ggml-dlcu/dl-moesum.cuh`

```cpp
#include "../ggml-cuda/common.cuh"

void ggml_cuda_op_moe_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

### 入口函数

**文件**: `ggml/src/ggml-dlcu/dl-moesum.cu:220-342`

```cpp
void ggml_cuda_op_moe_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // 获取输入张量
    ggml_tensor * src0 = dst->src[0];

    // 验证输入
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[0] == dst->ne[0]);
    GGML_ASSERT(src0->ne[2] == dst->ne[1]);

    // 解析形状
    const int token_num = src0->ne[2];      // T
    const int topk_num = src0->ne[1];       // K
    const int hidden_dim = src0->ne[0];     // D

    // 计算步长 (stride)
    const int stride_token = src0->nb[2] / src0->nb[0];   // 跨 token 步长
    const int stride_topk = src0->nb[1] / src0->nb[0];    // 跨专家步长
    const int out_stride_token = dst->nb[1] / dst->nb[0];

    auto stream = ctx.stream();

    // 路径选择策略
    // 1. 快速 FP16 向量化内核
    // 2. 小 token 数量内核
    // 3. warp-per-token 内核

    const bool fast_fp16_vec_ok = (src0->type == GGML_TYPE_F16) &&
                    (token_num > 256) && (hidden_dim % 8 == 0);
    if (fast_fp16_vec_ok) {
        // 路径 1: 向量化 FP16 内核
        // ...
    }

    const bool per_token_use_one_warp = (token_num > 128);
    if (!per_token_use_one_warp) {
        // 路径 2: 小 token 数量内核
        // ...
    } else {
        // 路径 3: warp-per-token 内核
        // ...
    }
}
```

### 路径选择决策树

```
                    moe_sum 内核选择
                            │
                            ▼
                ┌───────────────────────┐
                │ F16 && token > 256    │
                │ && hidden % 8 == 0    │
                └───────────────────────┘
                     │            │
                    Yes           No
                     │            │
                     ▼            ▼
        ┌────────────────┐   ┌──────────────────┐
        │ vec_kernel     │   │ token > 128 ?    │
        │ (128-bit loads)│   └──────────────────┘
        └────────────────┘        │        │
                                  Yes       No
                                   │        │
                                   ▼        ▼
                        ┌──────────────┐  ┌────────────┐
                        │warp_per_token│  │small_token │
                        │   kernel     │  │  kernel    │
                        └──────────────┘  └────────────┘
```

### 内核实现详解

#### 1. 向量化 FP16 内核 (最快路径)

**文件**: `ggml/src/ggml-dlcu/dl-moesum.cu:13-68`

```cpp
template <int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_vec_kernel(
    const half* __restrict__ x,
    half* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t topk_num,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token) {
  constexpr int VEC = 16;      // 每次处理 16 个元素 (128 bits)
  constexpr int PACKS = VEC / 8;  // 2 个 uint4

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int32_t t = blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  const int32_t n_chunks = hidden_dim / VEC;

  // 每个 lane 处理一个 chunk
  for (int32_t chunk = blockIdx.x * 32 + lane; chunk < n_chunks; chunk += gridDim.x * 32) {
    const int32_t d = chunk * VEC;
    const int32_t base = t * stride_token + d;

    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] = 0.f;

    // 对每个专家累加
#pragma unroll
    for (int k = 0; k < topk_num; ++k) {
#pragma unroll
      for (int p = 0; p < PACKS; ++p) {
        // 使用 128-bit 加载
        const int32_t offset = base + k * stride_topk + p * 8;
        Pack16B pack = {ldg_cg(reinterpret_cast<const uint4*>(x + offset))};

#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[p * 8 + i] += static_cast<float>(pack.u16[i]);
        }
      }
    }

    // 存储结果
#pragma unroll
    for (int p = 0; p < PACKS; ++p) {
      Pack16B outp;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        outp.u16[i] = static_cast<half>(acc[p * 8 + i]);
      }
      const int32_t dst = t * out_stride_token + d + p * 8;
      *reinterpret_cast<uint4*>(y + dst) = outp.v;
    }
  }
}
```

**优化技术**:

1. **128 位加载** (`Pack16B` / `uint4`)
   ```cpp
   union Pack16B {
     uint4 v;           // 128 bits = 16 bytes = 8 half values
     __half u16[8];
   };
   ```

2. **只读缓存** (`__ldg` / `ldg_cg`)
   ```cpp
   template <typename T>
   __device__ __forceinline__ T ldg_cg(const T* p) {
     return __ldg(p);  // 使用只读数据缓存 (LDG)
   }
   ```

3. **循环展开** (`#pragma unroll`)
   - 编译器完全展开内层循环
   - 减少分支开销

4. **寄存器累加**
   ```cpp
   float acc[VEC];  // 完全在寄存器中
   ```

**线程组织**:
```
Block: (WARPS_PER_BLOCK * 32) threads
       └── WARPS_PER_BLOCK warps

Grid: (grid_x, grid_y)
     ├── grid_x: chunks 并行度
     └── grid_y: token 分布

每个 warp 处理一个 token 的多个 chunks
```

#### 2. Warp-Per-Token 内核 (通用路径)

**文件**: `ggml/src/ggml-dlcu/dl-moesum.cu:70-95`

```cpp
template <typename scalar_t, int TOPK, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int32_t t = blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  // 每个 lane 处理 hidden_dim 的一部分
  for (int32_t d = blockIdx.x * 32 + lane; d < hidden_dim; d += gridDim.x * 32) {
    float acc = 0.f;
    const int32_t base = t * stride_token + d;

    // TOPK 是编译时常量，允许完全展开
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += static_cast<float>(x[base + k * stride_topk]);
    }

    y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
  }
}
```

**特点**:
- 每个 warp 处理一个 token
- warp 内 32 个 lane 并行处理 hidden_dim
- `TOPK` 作为模板参数，实现完全循环展开

**专用实例**: TOPK = 2, 4, 8, 9
```cpp
// 在启动时选择特化版本
if (topk_num == 2) {
    LAUNCH_WARP_PER_TOKEN_KERNEL(float, 2);
} else if (topk_num == 4) {
    LAUNCH_WARP_PER_TOKEN_KERNEL(float, 4);
} // ...
```

#### 3. 通用内核 (回退路径)

**文件**: `ggml/src/ggml-dlcu/dl-moesum.cu:97-122`

```cpp
template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_kernel_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token,
    const int topk_num) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int32_t t = blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int32_t d = blockIdx.x * 32 + lane; d < hidden_dim; d += gridDim.x * 32) {
    float acc = 0.f;
    const int32_t base = t * stride_token + d;
#pragma unroll 1  // 不展开，topk_num 是运行时值
    for (int k = 0; k < topk_num; ++k) {
      acc += static_cast<float>(x[base + k * stride_topk]);
    }

    y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
  }
}
```

**特点**:
- `topk_num` 作为运行时参数
- `#pragma unroll 1` 禁止循环展开
- 性能较特化版本略低，但支持任意 TOPK

#### 4. 小 Token 数量内核

**文件**: `ggml/src/ggml-dlcu/dl-moesum.cu:124-146`

```cpp
template <typename scalar_t, int TOPK>
__global__ void moe_sum_reduce_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int32_t token_num,
    const int32_t hidden_dim,
    const int32_t stride_token,
    const int32_t stride_topk,
    const int32_t out_stride_token) {
  // 2D 网格: blockIdx.x 处理 hidden_dim, blockIdx.y 处理 token
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int32_t base = t * stride_token + d;
      float acc = 0.f;

#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += static_cast<float>(x[base + k * stride_topk]);
      }

      y[t * out_stride_token + d] = static_cast<scalar_t>(acc);
    }
  }
}
```

**使用场景**: `token_num <= 128`
- 使用 2D 网格布局
- 适合小批量推理

### 启动宏定义

```cpp
#define LAUNCH_SMALL_TOKEN_KERNEL(scalar_t, TOPK)                       \
    moe_sum_reduce_kernel<scalar_t, TOPK><<<grid, block, 0, stream>>>(  \
        static_cast<scalar_t*>(src0->data),                             \
        static_cast<scalar_t*>(dst->data),                              \
        token_num,                                                      \
        hidden_dim,                                                     \
        stride_token,                                                   \
        stride_topk,                                                    \
        out_stride_token);

#define LAUNCH_WARP_PER_TOKEN_KERNEL(scalar_t, TOPK)                    \
    moe_sum_reduce_warp_token_kernel<scalar_t, TOPK, WARPS_PER_BLOCK>   \
            <<<grid, block, 0, stream>>>(                               \
        static_cast<scalar_t*>(src0->data),                             \
        static_cast<scalar_t*>(dst->data),                              \
        token_num,                                                      \
        hidden_dim,                                                     \
        stride_token,                                                   \
        stride_topk,                                                    \
        out_stride_token);
```

---

## 调度机制

### CPU 调度

**文件**: `ggml/src/ggml-cpu/ggml-cpu.c`

**case 分发** (`:2101-2105`):
```cpp
case GGML_OP_MOE_SUM:
    {
        ggml_compute_forward_moe_sum(params, tensor);
    } break;
```

**任务分配** (`:2449-2453`):
```cpp
case GGML_OP_MOE_SUM:
    {
        n_tasks = n_threads;  // 允许多线程并行
    } break;
```

### CUDA 调度

**文件**: `ggml/src/ggml-cuda/ggml-cuda.cu`

**case 分发** (`:2865-2868`):
```cpp
case GGML_OP_MOE_SUM:
    ggml_cuda_op_moe_sum(ctx, dst);
    break;
```

**支持检查** (`:4939-4941`):
```cpp
case GGML_OP_MOE_SUM:
    return true;  // CUDA 支持此操作
```

### 完整调度流程

```
用户调用 ggml_moe_sum()
         │
         ▼
构建计算图 (graph build)
         │
         ▼
ggml_graph_compute()
         │
         ├──▶ CPU 模式
         │      │
         │      ▼
         │   ggml-cpu.c: switch(op)
         │      │
         │      ▼
         │   ggml_compute_forward_moe_sum()
         │
         └──▶ CUDA 模式
                │
                ▼
             ggml-cuda.cu: switch(op)
                │
                ▼
             ggml_cuda_op_moe_sum()
                │
                ├──▶ 选择内核策略
                │      │
                │      ├── vec_kernel (F16 + 大 token)
                │      ├── small_token_kernel
                │      ├── warp_per_token_kernel (固定 TOPK)
                │      └── general_kernel (任意 TOPK)
                │
                ▼
             启动 CUDA kernel
```

---

## 性能分析与优化

### 性能特征

| 内核类型 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| `vec_kernel` | F16 + token>256 + hidden%8==0 | 128-bit 加载，最高吞吐 | 条件严格 |
| `warp_per_token_kernel` | token>128, 固定 TOPK | 完全展开，高效率 | 仅限特定 TOPK |
| `general_kernel` | token>128, 任意 TOPK | 灵活 | 循环未展开 |
| `small_token_kernel` | token<=128 | 小批量友好 | 全局内存访问多 |

### 内存访问模式分析

**向量化加载优势**:
```
标量加载 (8x):
[1][2][3][4][5][6][7][8]
 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
8 次事务

128-bit 加载 (1x):
[1|2|3|4|5|6|7|8]
        ↓
1 次事务 (16 bytes)
```

**对齐要求**:
- `hidden_dim % 8 == 0` 确保边界对齐
- `Pack16B` 需要 16 字节对齐

### 计算强度分析

**算术强度** (Arithmetic Intensity):
```
每个元素操作数:
- 读取: topk_num 次
- 累加: topk_num - 1 次
- 写回: 1 次

总操作: O(D * T * K)
总数据: O(D * T * K)
```

**优化策略**:
1. **寄存器复用** - 减少 GLD/GST 次数
2. **循环展开** - 隐藏延迟
3. **向量化** - 提高带宽利用率

### 不同配置性能预测

```
配置 A: D=4096, K=4, T=512
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vec_kernel:       ████████████████████ 100%
warp_token:       ████████████████  85%
general:          ██████████████    75%

配置 B: D=4096, K=8, T=128
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vec_kernel:       ████████████████████ 100%
warp_token:       ████████████████  88%
general:          ██████████████    78%

配置 C: D=2048, K=2, T=32
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
small_token:      ████████████████████ 100%
warp_token:       ████████████       70%
```

---

## 完整调用链

### 图构建阶段

```
llama_model_load_internal()
  │
  └──▶ MoE 层构建
        │
        └──▶ ggml_mul_mat_id()  [专家计算]
              │
              └──▶ ggml_moe_sum()  [专家合并]
                    │
                    └──▶ 返回计算图节点
```

### 计算阶段 (CPU)

```
ggml_graph_compute_with_ctx()
  │
  └──▶ ggml_graph_compute()
        │
        └──▶ ggml_cuda_op_moe_sum()  [如果可用]
              │
              └──▶ ggml_compute_forward_moe_sum()
                    │
                    ├──▶ memset(dst, 0)
                    │
                    └──▶ for each expert
                          │
                          └──▶ ggml_compute_forward_add()
```

### 计算阶段 (CUDA/DLCU)

```
ggml_graph_compute_with_ctx()
  │
  └──▶ ggml_graph_compute()
        │
        └──▶ CUDA backend 调度
              │
              └──▶ ggml_cuda_op_moe_sum()
                    │
                    ├──▶ 检查输入类型和形状
                    │
                    ├──▶ 选择内核路径
                    │      │
                    │      ├──▶ vec_kernel (F16 + 大 token)
                    │      ├──▶ small_token_kernel
                    │      ├──▶ warp_per_token_kernel
                    │      └──▶ general_kernel
                    │
                    └──▶ cudaLaunchKernel()
                          │
                          └──▶ GPU 执行
```

### 模型使用示例

**文件**: `src/llama-model.cpp:211-216`

```cpp
case GGML_OP_MOE_SUM:
    {
        int n_expert_used = hparams.n_expert_used;
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                              w->ne[0], n_expert_used, 512);
        op_tensor = ggml_moe_sum(ctx, a, n_expert_used);
    } break;
```

---

## 调试与验证

### CPU vs GPU 正确性验证

```cpp
// 在测试代码中比较 CPU 和 GPU 输出
ggml_graph_cpu_compute(graph_cpu);
ggml_graph_cuda_compute(graph_gpu);

for (int i = 0; i < nelements; ++i) {
    assert(fabs(cpu_output[i] - gpu_output[i]) < 1e-5);
}
```

### 常见问题诊断

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 性能低于预期 | 内核选择不匹配 | 检查 `token_num` 和 `hidden_dim` |
| 数值不正确 | stride 计算错误 | 验证 `nb[]` 值 |
| 内存错误 | 张量不连续 | 确保 `ggml_is_contiguous()` |

---

## 参考资料

- [小白版入门文档](../index.md)
- [MoE 架构说明](../../../docs/for-me/moe-architecture.md)
- [Top-K 专家选择](../../../docs/for-me/Top-K-Expert-Selection.md)

---

**文档版本**: 1.0
**最后更新**: 2026-01-30
**维护者**: llama.cpp 社区
