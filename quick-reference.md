# Prefix Cache 快速参考

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache-reuse N` | 32 | Cache Reuse阈值，连续匹配>=N tokens时复用 |
| `--cache-prompt-size` | 8192 MiB | Global Cache容量限制 |
| `--cache-prompt-tokens` | 0 | Global Cache token数量限制(0=无限制) |
| `--n-slots` | 8 | Slot数量 |

## API参数

```json
{
    "cache_prompt": true,  // 默认true，启用prompt caching
    "id_slot": 0           // 可选，指定使用特定slot
}
```

## 关键代码位置速查

| 功能 | 文件:行号 | 关键代码 |
|------|----------|---------|
| LCP计算 | server-context.cpp:2122 | `get_common_prefix(input_tokens)` |
| Cache Reuse | server-context.cpp:2141-2188 | KV cache shifting |
| 截断 | server-context.cpp:2317 | `keep_first(n_past)` |
| Global Cache加载 | server-task.cpp:1432-1482 | `prompt = std::move(*it_best)` |
| Global Cache淘汰 | server-task.cpp:1484-1524 | `states.pop_front()` |

## cache_prompt行为

| cache_prompt | n_past | slot.prompt.tokens | 效果 |
|-------------|--------|-------------------|------|
| true | LCP长度 | 截断为LCP长度 | 启用caching |
| false | 0 | 清空 | 禁用caching |

## cache_reuse条件

1. `n_cache_reuse > 0`
2. `can_cache_reuse = true`
3. 连续匹配tokens >= n_cache_reuse

## Global Cache命中条件

1. LCP相似度 >= 25% (`f_keep >= 0.25`)
2. 相似度和保留率都优于当前最佳

## 常见场景TTFT

| 场景 | 首次请求 | 后续请求 | 加速比 |
|------|---------|---------|--------|
| 精确匹配 | ~2000ms | ~50ms | 40x |
| 相似前缀(增量>=32) | ~2000ms | ~500ms | 4x |
| 相似前缀(增量<32) | ~2000ms | ~2200ms | 1x |
| 完全不同 | ~2000ms | ~2000ms |

## 故障 | 1x排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| cache_reuse不工作 | 增量<32tokens | 确保增量>=32或降低阈值 |
| Global Cache未命中 | 被LRU淘汰 | 增加cache-prompt-size |
| TTFT没有改善 | cache_prompt=false | 确认参数为true |
| 多模态场景cache_reuse不工作 | 不支持多模态 | 这是预期行为 |

## 优化建议

1. **发送完整对话历史** - 确保每次请求包含完整context
2. **确保增量足够长** - >=32tokens以触发cache_reuse
3. **使用固定slot** - 通过id_slot参数
4. **增加Global Cache容量** - --cache-prompt-size 16384
5. **启用调试** - --slots-debug 查看详细信息

