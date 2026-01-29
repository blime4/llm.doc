# LLM.DOC

结构化LLM学习笔记仓库，按知识点组织。

## 知识点

### KV Cache复用

| 框架 | 主题 | 内容 |
|------|------|------|
| [llama.cpp](llama.cpp/prefix-cache/) | Prefix Cache | 两层缓存架构、Slot-level/Global Cache |
| [vLLM](vllm/prefix-cache/) | Zero-Overhead Prefix Cache | 零开销实现、结构/操作/代码优化 |

## 框架

### llama.cpp

- [prefix-cache](llama.cpp/prefix-cache/) - KV Cache复用机制

### vLLM

- [prefix-cache](vllm/prefix-cache/) - 零开销前置缓存

## 使用

```bash
# 作为submodule添加到Obsidian
git submodule add git@github.com:blime4/llm.doc.git <target-dir>
```
