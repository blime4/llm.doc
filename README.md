# LLM.DOC

结构化LLM学习笔记仓库，按知识点组织。

## 知识点

### Prefix Cache

KV Cache跨请求复用机制，减少prefill阶段计算量。

| 框架 | 实现 |
|------|------|
| [llama.cpp](llama.cpp/prefix-cache/) | 两层缓存架构：Slot-level Cache Reuse + Global Prompt Cache |
| [vLLM](vllm/prefix-cache/) | 零开销Prefix Cache：结构/操作/代码优化 |

### MoE (混合专家)

混合专家模型相关算子实现。

| 算子 | 描述 |
|------|------|
| [moe-sum](llama.cpp/moe-sum/) | MoE 专家输出聚合算子 |

## 框架

### llama.cpp

- [prefix-cache](llama.cpp/prefix-cache/) - KV Cache复用机制
- [moe-sum](llama.cpp/moe-sum/) - MoE 专家输出聚合算子

### vLLM

- [prefix-cache](vllm/prefix-cache/) - 零开销前置缓存

## 使用

```bash
# 作为submodule添加到Obsidian
git submodule add git@github.com:blime4/llm.doc.git <target-dir>
```
