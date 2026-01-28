# Prefix Cache 图解集

本文档包含使用Mermaid语法绘制的图表，可以在支持Mermaid的编辑器中查看。

## 1. 两层缓存架构

```mermaid
flowchart TB
    subgraph Client["客户端请求"]
        Request["新请求"]
    end

    subgraph Slots["多个Slots"]
        Slot0["Slot 0\nprompt.tokens\nKV Cache"]
        Slot1["Slot 1\nprompt.tokens\nKV Cache"]
        Slot2["Slot 2\nprompt.tokens\nKV Cache"]
    end

    subgraph Global["Global Prompt Cache"]
        Cache["存储: 完整Prompt State\n策略: LRU淘汰\n容量: 8192 MiB"]
    end

    Request --> SlotSelection["Slot Selection"]
    SlotSelection --> Slot0
    SlotSelection --> Slot1
    SlotSelection --> Slot2

    Slot0 -.-> Global
    Slot1 -.-> Global
    Slot2 -.-> Global

    style Global fill:#e1f5fe
    style Slots fill:#f3e5f5
```

## 2. 数据流向

```mermaid
flowchart LR
    A["新请求到达"] --> B["Global Cache查找"]
    B --> C{命中?}
    C -->|是| D["完全替换slot.prompt"]
    C -->|否| E["计算LCP"]
    D --> F["处理新tokens"]
    E --> F
    F --> G["保存到Global Cache\n(slot释放时)"]

    style A fill:#ffeb3b
    style G fill:#c8e6c9
```

## 3. slot.prompt.tokens生命周期

```mermaid
flowchart TB
    subgraph Init["初始化"]
        Empty["[]"]
    end

    subgraph Req1["请求1: base prompt"]
        LCP1["LCP=0"]
        Trunc1["keep_first(0) → []"]
        Process1["处理base_1...base_1000"]
        Gen1["生成gen_1"]
        Result1["[base_1...base_1000, gen_1]"]
    end

    subgraph Req2["请求2: base + ext1"]
        LCP2["LCP=1000\ngen_1 != ext1_1"]
        Trunc2["keep_first(1000)\n丢弃generated tokens"]
        Process2["处理ext1_1, ext1_2, ext1_3"]
        Result2["[base_1...base_1000,\next1_1, ext1_2, ext1_3]"]
    end

    subgraph Req3["请求3: cache_prompt=false"]
        Trunc3["keep_first(0)\n全部清空"]
        Process3["重新处理整个prompt"]
    end

    Init --> Req1
    Req1 --> Req2
    Req2 --> Req3

    style Init fill:#e0e0e0
    style Req3 fill:#ffcdd2
```

## 4. LCP计算

```mermaid
flowchart LR
    subgraph Old["旧缓存"]
        OldTokens["A B C D E F G"]
    end

    subgraph New["新请求"]
        NewTokens["A B C D E F G H I J"]
    end

    subgraph Result["LCP结果"]
        LCP["LCP = 7\n(ABCDEFG)"]
    end

    OldTokens --> Compare["逐个比较"]
    NewTokens --> Compare
    Compare --> Result

    style Result fill:#c8e6c9
```

## 5. Cache Reuse机制

```mermaid
flowchart TB
    subgraph Cache["旧KV Cache"]
        KV0["[A B C D E F G H]"]
    end

    subgraph Input["新请求"]
        Input0["[A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f]"]
    end

    subgraph LCP["计算LCP"]
        LCP0["n_past = 18\n(XYZ...ef)"]
    end

    subgraph Search["搜索匹配chunk"]
        Search0["从位置18开始搜索\n旧缓存只有18个tokens\n搜索失败"]
    end

    KV0 --> Input0
    Input0 --> LCP0
    LCP0 --> Search0

    style Search0 fill:#ffcdd2
```

## 6. Global Cache LRU

```mermaid
flowchart LR
    subgraph Before["淘汰前"]
        State0["state0: [A B C D E]\n200 MiB ★最老"]
        State1["state1: [F G H I J]\n200 MiB"]
        State2["state2: [K L M N O]\n200 MiB"]
        State3["state3: [P Q R S T]\n200 MiB ★最新"]
    end

    subgraph After["淘汰后"]
        State1_2["state1: [F G H I J]\n200 MiB ★最老"]
        State2_2["state2: [K L M N O]\n200 MiB"]
        State3_2["state3: [P Q R S T]\n200 MiB"]
        NewState["state4: [U V W X Y]\n200 MiB ★最新"]
    end

    Before -->|pop_front()| After
    NewState --> After

    style Before fill:#fff3e0
    style After fill:#e8f5e9
```

## 7. cache_prompt对比

```mermaid
flowchart TB
    subgraph True["cache_prompt=true"]
        R1a["请求1: 处理整个prompt\nTTFT=2000ms"]
        R2a["请求2: 跳过处理\nTTFT=50ms ✓"]
    end

    subgraph False["cache_prompt=false"]
        R1b["请求1: 处理整个prompt\nTTFT=2000ms"]
        R2b["请求2: 重新处理\nTTFT=2000ms ✗"]
    end

    R1a --> R2a
    R1b --> R2b

    style True fill:#c8e6c9
    style False fill:#ffcdd2
```

## 8. Multi-modal处理

```mermaid
flowchart TB
    subgraph Tokens["tokens数组"]
        T1["[T1]"]
        T2["[T2]"]
        T3["[NULL]"]
        T4["[NULL]"]
        T5["[NULL]"]
        T6["[T6]"]
        T7["[T7]"]
    end

    subgraph Map["map_idx_to_media"]
        Map1["index 2 → ImageChunk\n(id='img_001', n=3)"]
    end

    subgraph Trunc["截断规则"]
        Allow["✓ 截断位置5\n完整保留image"]
        Disallow["✗ 截断位置3\n在image中间截断"]
    end

    Tokens --> Map
    Map --> Trunc

    style Allow fill:#c8e6c9
    style Disallow fill:#ffcdd2
```

## 9. 增长前缀测试分析

```mermaid
gantt
    title 增长前缀请求时序
    dateFormat X
    axisFormat %s

    section 请求1
    处理base (1000tokens) :0, 2106
    TTFT:2106ms         :crit, 2106, 2106

    section 请求2
    LCP计算(1000)       :2106, 2110, 2115
    cache_reuse搜索     :2115, 2125, 2200
    处理ext1(3tokens)   :2200, 2280, 2290
    TTFT:2290ms         :crit, 2290, 2290

    section 请求3
    Global Cache命中    :2290, 2300, 2330
    恢复KV Cache        :2330, 2350, 2400
    TTFT:256ms ✓        :crit, 2546, 2546

    section 请求4
    Global Cache未命中   :2546, 2550, 2600
    重新处理           :2600, 4500, 4647
    TTFT:2101ms ✗       :crit, 4647, 4647
```

## 10. 完整处理流程

```mermaid
flowchart TD
    Start["新请求到达"] --> Global["1. Global Cache查找\nprompt_load()"]
    Global --> CacheHit{命中?}
    CacheHit -->|是| Replace["2. 完全替换\nslot.prompt = std::move()"]
    CacheHit -->|否| LCP["3. 计算LCP\nget_common_prefix()"]
    
    LCP --> Split{can_split?}
    Split -->|否| Trunc["截断: keep_first(0)"]
    Split -->|是| CachePrompt{cache_prompt?}
    
    CachePrompt -->|否| Trunc2["截断: keep_first(0)"]
    CachePrompt -->|是| Reuse["4. Cache Reuse\nKV shifting (可选)"]
    
    Reuse --> Trunc3["5. 截断\nkeep_first(n_past)"]
    Trunc --> Process["6. 处理新tokens\npush_back()"]
    Trunc2 --> Process
    Trunc3 --> Process
    
    Process --> Done["7. 生成完成\nslot释放"]
    
    Replace --> Done

    style CacheHit fill:#fff3e0
    style CachePrompt fill:#e3f2fd
    style Done fill:#c8e6c9
```

## 11. 两层机制对比

```mermaid
flowchart LR
    subgraph Layer1["第一层: Slot-level Cache Reuse"]
        S1["作用范围: 同一slot"]
        M1["机制: KV shifting"]
        T1["阈值: >=32 tokens"]
        E1["效果: 部分KV复用"]
        L1["Multi-modal: 不支持"]
    end

    subgraph Layer2["第二层: Global Prompt Cache"]
        S2["作用范围: 跨所有slots"]
        M2["机制: State替换"]
        T2["阈值: LCP相似度>=25%"]
        E2["效果: 完整State替换"]
        L2["Multi-modal: 支持"]
    end

    Layer1 -.->|互补| Layer2

    style Layer1 fill:#f3e5f5
    style Layer2 fill:#e1f5fe
```

## 12. 性能对比

```mermaid
xychart-beta
    title TTFT对比 (ms)
    x-axis [精确R1, 精确R2, 增长R1, 增长R2, 增长R3, 增长R4]
    y-axis 0 --> 2500
    bar [2225, 42, 2106, 2290, 256, 2101]
```

