# Prefix Cache 文档索引

## 文档结构

```
prefix-cache/
├── README.md              # 主文档(完整版)
├── diagrams.md            # Mermaid图解集
├── quick-reference.md     # 快速参考指南
├── test-cases.md          # 测试用例详解
└── index.md               # 本索引
```

## 快速开始

1. **新手入门**: 阅读 [README.md](README.md) 的第1-2章
2. **理解原理**: 阅读 [README.md](README.md) 的第3-7章
3. **查看图解**: 查看 [diagrams.md](diagrams.md)
4. **快速参考**: 使用 [quick-reference.md](quick-reference.md)
5. **深入分析**: 阅读完整 [README.md](README.md)

## 内容概览

### README.md (主文档)
- [1. 概述](README.md#1-概述) - 什么是Prefix Cache
- [2. 两层缓存架构](README.md#2-两层缓存架构) - Slot-level vs Global
- [3. 核心数据结构](README.md#3-核心数据结构) - server_tokens, server_prompt
- [4. slot.prompt.tokens生命周期](README.md#4-slotprompttokens生命周期) - 截断机制详解
- [5. LCP计算](README.md#5-lcp-最长公共前缀计算) - 最长公共前缀算法
- [6. Slot-level Cache Reuse](README.md#6-第一层slotlevel-cache-reuse) - KV cache shifting
- [7. Global Prompt Cache](README.md#7-第二层global-prompt-cache) - 跨slot缓存和LRU
- [8. cache_prompt参数](README.md#8-cache_prompt参数) - 参数作用和效果
- [9. 多模态场景](README.md#9-多模态场景处理) - Multi-modal限制和保护
- [10. 测试数据分析](README.md#10-测试数据分析) - 测试结果和原因分析
- [11. FAQ](README.md#11-常见问题与解决方案) - 常见问题和调试技巧
- [12. 最佳实践](README.md#12-最佳实践) - 优化建议
- [13. 代码位置](README.md#13-关键代码位置) - 关键代码行号汇总

### diagrams.md (图解集)
- 两层缓存架构图
- 数据流向图
- 生命周期流程图
- LCP计算示例图
- Cache Reuse机制图
- Global Cache LRU图
- cache_prompt对比图
- Multi-modal处理图
- 增长前缀测试时序图
- 完整处理流程图
- 性能对比图

### quick-reference.md (快速参考)
- 核心参数速查表
- API参数说明
- 代码位置速查
- 常见场景TTFT参考
- 故障排查指南
- 优化建议清单

## 关键概念

### 两层缓存

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   第一层: Slot-level Cache Reuse                     │
│   • 同一slot内复用                                   │
│   • KV cache shifting                               │
│   • 阈值: >=32 tokens                                │
│                                                     │
│   第二层: Global Prompt Cache                        │
│   • 跨slot共享                                       │
│   • 完整state替换                                    │
│   • LRU淘汰                                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 核心机制

1. **截断机制**: `slot.prompt.tokens` 不会无限增长，每次请求被截断为LCP长度
2. **cache_prompt**: 控制是否启用prompt caching
3. **cache_reuse**: 复用>=32tokens的连续匹配chunk
4. **Global Cache**: 跨slot存储完整prompt state

## 相关资源

- 源代码: `tools/server/server-context.cpp`
- 源代码: `tools/server/server-task.cpp`
- 源代码: `tools/server/server-common.cpp`
- 测试脚本: `tests/test_prefix_cache.sh`
- 分析文档: `docs/for-me/llama-cpp-prefix-cache-deep-dive.md`

## 使用建议

### 日常开发
1. 遇到问题先查 [quick-reference.md](quick-reference.md)
2. 需要深入理解时阅读 [README.md](README.md)
3. 调试时查看 [diagrams.md](diagrams.md)

### 性能优化
1. 确保增量>=32tokens
2. 发送完整对话历史
3. 使用固定slot
4. 增大Global Cache容量

