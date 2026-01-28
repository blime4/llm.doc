# llama.cpp Prefix Cache 完整指南

## 目录
1. [概述](#1-概述)
2. [两层缓存架构](#2-两层缓存架构)
3. [核心数据结构](#3-核心数据结构)
4. [slot.prompt.tokens生命周期](#4-slotprompttokens生命周期)
5. [LCP最长公共前缀计算](#5-lcp最长公共前缀计算)
6. [第一层:Slot-levelCacheReuse](#6-第一层slotlevel-cache-reuse)
7. [第二层:GlobalPromptCache](#7-第二层global-prompt-cache)
8. [cache_prompt参数](#8-cache_prompt参数)
9. [多模态场景处理](#9-多模态场景处理)
10. [测试数据分析](#10-测试数据分析)
11. [常见问题与解决方案](#11-常见问题与解决方案)
12. [最佳实践](#12-最佳实践)
13. [关键代码位置](#13-关键代码位置)

## 1. 概述

### 1.1 什么是PrefixCache？

PrefixCache是llama.cpp中的**KVCache复用机制**，用于在处理相似prompt时复用之前计算过的Key-Value向量，从而大幅减少计算量，降低首token时间(TTFT)。

```
+------------------------------------------------------------------+
|                    PrefixCache的核心价值                           |
+------------------------------------------------------------------+
|                                                                  |
|   第一次请求:  Prompt[A B C D E F G H I J K L...]               |
|                计算所有KVCache                                    |
|                TTFT:2000ms                                       |
|                                                                  |
|   第二次请求:  Prompt[A B C D E F G H I J K L...+新内容]         |
|                前缀相同                                           |
|                复用已有KVCache                                    |
|                TTFT:50ms(98%加速!)                               |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 为什么需要PrefixCache？

在对话系统中，连续的请求往往共享相同的上下文：

```json
// 请求1
{"messages":[{"role":"user","content":"请写一篇关于AI的文章..."}]}

// 请求2
{"messages":[{"role":"user","content":"请写一篇关于AI的文章..."},{"role":"assistant","content":"以下是文章..."},{"role":"user","content":"继续"}]}
```

请求2的prompt包含了请求1的所有内容，如果没有prefixcache，需要重新计算所有KV值。

### 1.3 性能对比

| 场景 | 无Cache | 有Cache | 加速比 |
|------|---------|---------|--------|
| 精确匹配请求 | 2000ms | 40ms | 50x |
| 相似前缀请求 | 2000ms | 500ms | 4x |
| 完全不同请求 | 2000ms | 2000ms | 1x |

## 2. 两层缓存架构

llama.cpp的prefixcache分为**两层独立机制**，互为补充：

```
+----------------------------------------------------------------------+
|                        PrefixCache两层架构                             |
+----------------------------------------------------------------------+
|                                                                      |
|   +-------------------------------------------------------------+    |
|   |                    客户端请求流程                            |    |
|   +-------------------------------------------------------------+    |
|                               |                                      |
|                               v                                      |
|   +-------------------------------------------------------------+    |
|   |              SlotSelection(选择处理请求的slot)               |    |
|   +-------------------------------------------------------------+    |
|                               |                                      |
|              +----------------+----------------+                     |
|              |                 |                |                     |
|              v                 v                v                     |
|      +-------------+   +-------------+   +-------------+            |
|      |  Slot0      |   |  Slot1      |   |  Slot2      |            |
|      | prompt.tokens|   | prompt.tokens|   | prompt.tokens|            |
|      | KVCache     |   | KVCache     |   | KVCache     |            |
|      +-------------+   +-------------+   +-------------+            |
|              |                 |                |                   |
|              +----------------+----------------+                    |
|                               |                                      |
|                               v                                      |
|   +-------------------------------------------------------------+    |
|   |              GlobalPromptCache(跨slots共享)                  |    |
|   |              容量:8192MiB(默认)                              |    |
|   |              策略:LRU淘汰                                    |    |
|   |              存储:完整promptstate                            |    |
|   +-------------------------------------------------------------+    |
|                                                                      |
+----------------------------------------------------------------------+
```

### 2.1 两层机制对比

| 特性 | 第一层:Slot-levelCacheReuse | 第二层:GlobalPromptCache |
|------|-------------------------------|--------------------------|
| **作用范围** | 同一slot | 跨所有slots |
| **机制** | KVCache位置偏移(shifting) | 完整State替换 |
| **阈值** | --cache-reuseN(默认32) | LCP相似度>=25% |
| **效果** | 部分KVCache重用 | 完整State替换 |
| **Multi-modal** | 不支持 | 支持 |
| **容量限制** | 无 | 8192MiB+Token数量限制 |
| **淘汰策略** | 无(随slot释放) | LRU |
| **依赖参数** | cache_prompt=true | 独立工作 |

### 2.2 数据流向图

```
+---------------------------------------------------------------------+
|                           请求处理数据流向                            |
+---------------------------------------------------------------------+
|                                                                     |
|  新请求到达                                                          |
|      |                                                             |
|      v                                                             |
|  +--------------------------------------------------------------+   |
|  | 1.GlobalCache查找                                            |   |
|  |     prompt_load()                                            |   |
|  |     跨slot搜索相似prompt                                     |   |
|  |     命中则完全替换slot.prompt                                |   |
|  +--------------------------------------------------------------+   |
|      |                                                             |
|      | 未命中                                                       |
|      v                                                             |
|  +--------------------------------------------------------------+   |
|  | 2.计算LCP                                                    |   |
|  |     get_common_prefix()                                      |   |
|  |     比较slot.prompt.tokens和input_tokens                     |   |
|  |     计算最长公共前缀长度                                      |   |
|  +--------------------------------------------------------------+   |
|      |                                                             |
|      v                                                             |
|  +--------------------------------------------------------------+   |
|  | 3.Slot-levelCacheReuse(可选)                                  |   |
|  |     搜索LCP之后的连续匹配chunk                                |   |
|  |     满足n_cache_reuse阈值才复用                               |   |
|  |     KVcachesifting                                           |   |
|  +--------------------------------------------------------------+   |
|      |                                                             |
|      v                                                             |
|  +--------------------------------------------------------------+   |
|  | 4.截断slot.prompt.tokens                                      |   |
|  |     keep_first(n_past)                                       |   |
|  |     只保留LCP部分                                             |   |
|  |     丢弃非LCP部分的旧tokens                                   |   |
|  +--------------------------------------------------------------+   |
|      |                                                             |
|      v                                                             |
|  +--------------------------------------------------------------+   |
|  | 5.处理新tokens                                               |   |
|  |     push_back()新token                                       |   |
|  |     计算新增KVCache                                           |   |
|  +--------------------------------------------------------------+   |
|      |                                                             |
|      v                                                             |
|  +--------------------------------------------------------------+   |
|  | 6.保存到GlobalCache(slot释放时)                               |   |
|  |     prompt_save()                                            |   |
|  |     存储完整promptstate                                      |   |
|  |     LRU淘汰最老entry                                          |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+
```

## 3. 核心数据结构

### 3.1 server_tokens结构

```cpp
struct server_tokens {
    bool has_mtmd = false;  //是否包含多模态数据

private:
    std::map<size_t,mtmd::input_chunk_ptr> map_idx_to_media;  //mediachunk映射
    llama_tokens tokens;  //token列表

public:
    void push_back(llama_token tok);           //添加token
    void keep_first(size_t n);                 //截断到前ntokens
    size_t get_common_prefix(const server_tokens & b) const;  //计算LCP
    void set_token(size_t pos,llama_token id);  //设置指定位置token
    size_t size() const;                      //返回token数量
};
```

### 3.2 数据结构内部图解

```
+---------------------------------------------------------------------+
|                         server_tokens内部结构                         |
+---------------------------------------------------------------------+
|                                                                     |
|   tokens数组(示例):                                                 |
|   +-----+-----+-----+-----+-----+---------+---------+-----+-----+  |
|   | 101 | 234 | 892 | 105 | 672 |  NULL   |  NULL   | 445 | 891 |  |
|   +-----+-----+-----+-----+-----+---------+---------+-----+-----+  |
|     0     1     2     3     4     5        6        7     8         |
|                                   |                                  |
|                                   +--LLAMA_TOKEN_NULL                |
|                                       (mediachunk占位符)            |
|                                                                     |
|   map_idx_to_media映射:                                            |
|   +----------------------------------------------------------+      |
|   |  index5->ImageChunk(id="img_001",n_tokens=3)            |      |
|   |  index8->AudioChunk(id="audio_002",n_tokens=2)          |      |
|   +----------------------------------------------------------+      |
|                                                                     |
|   has_mtmd=true表示包含多模态内容                                   |
|                                                                     |
+---------------------------------------------------------------------+
```

### 3.3 server_prompt结构

```cpp
struct server_prompt {
    server_tokens tokens;           //token列表
    std::vector<uint8_t> data;      //KVcachestate数据
    std::vector<server_prompt_checkpoint> checkpoints;  //SWAcheckpoint
    size_t size() const;            //数据大小
    size_t n_tokens() const;        //token数量
};
```

### 3.4 GlobalCache结构

```cpp
class server_prompt_cache {
private:
    std::list<server_prompt> states;  //LRU列表
    size_t limit_size;                //容量限制(bytes)
    size_t limit_tokens;              //token数量限制

public:
    server_prompt * alloc(const server_prompt & prompt,size_t state_size);
    bool load(server_prompt & prompt,const server_tokens & tokens_new,
              llama_context * ctx,int32_t id_slot);
    void update();  //LRU淘汰
};
```

```
+---------------------------------------------------------------------+
|                     GlobalPromptCache结构                            |
+---------------------------------------------------------------------+
|                                                                     |
|   states(LRU列表):                                                  |
|                                                                     |
|   +-----------------------------------------------------------+     |
|   |  +----------+  +----------+  +----------+  +----------+  |     |
|   |  | state0  |<->| state1  |<->| state2  |<->| state3  |  |     |
|   |  | (最老)   |  |          |  |          |  | (最新)   |  |     |
|   |  +----------+  +----------+  +----------+  +----------+  |     |
|   |      ^                                      ^           |     |
|   |      |                                      |           |     |
|   |   pop_front()                           push_back()    |     |
|   |   (LRU淘汰)                             (新插入)        |     |
|   +-----------------------------------------------------------+     |
|                                                                     |
|   每个state包含:                                                      |
|   -tokens:server_tokens(prompt的token列表)                           |
|   -data:std::vector<uint8_t>(KVcachestate)                          |
|   -checkpoints:SWAcheckpoint列表                                     |
|                                                                     |
+---------------------------------------------------------------------+
```

## 4. slot.prompt.tokens生命周期

### 4.1 关键发现:截断机制

**重要**:`slot.prompt.tokens`并**不会无限增长**！每次新请求到达时都会被截断。

```cpp
//server-context.cpp:2122-2317
n_past=slot.prompt.tokens.get_common_prefix(input_tokens);  //计算LCP

//...cachereuse处理...

slot.prompt.tokens.keep_first(n_past);  //关键:只保留前n_past个tokens！
```

### 4.2 keep_first实现

```cpp
void server_tokens::keep_first(size_t n) {
    GGML_ASSERT(n<=tokens.size());

    if(has_mtmd){
        if(n==tokens.size()){
            return;  //无变化
        }

        //保护:不能在image中间截断
        if(n>0){
            if(tokens[n-1]==LLAMA_TOKEN_NULL&&tokens[n]==LLAMA_TOKEN_NULL){
                find_chunk(n-1);  //抛出错误(不是chunk起始位置)
            }
        }

        //清理不再使用的imagechunks
        for(auto it=map_idx_to_media.begin();it!=map_idx_to_media.end();){
            if(it->first>=n){
                it=map_idx_to_media.erase(it);
            }else{
                ++it;
            }
        }
    }

    tokens.resize(n);  //截断token数组
}
```

### 4.3 完整生命周期图解

```
+---------------------------------------------------------------------+
|                    slot.prompt.tokens完整生命周期                     |
+---------------------------------------------------------------------+
|                                                                     |
|  [初始化]                                                           |
|    slot.prompt.tokens=[]                                            |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [请求1:baseprompt(1000texttokens),cache_prompt=true]               |
|                                                                     |
|    原始:     []                                                     |
|    LCP:      LCP([],[base])=0                                      |
|    截断:     keep_first(0)->[]                                      |
|    处理:     push_back(base_1)...push_back(base_1000)               |
|    生成:     push_back(gen_1)                                       |
|    结果:     [base_1,base_2,...,base_1000,gen_1]                     |
|    释放:     slot.release()->保留状态                               |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [请求2:base+ext1(1003tokens,ext1=3tokens),cache_prompt=true]       |
|                                                                     |
|    旧状态:   [base_1,...,base_1000,gen_1]                           |
|    新请求:   [base_1,...,base_1000,ext1_1,ext1_2,ext1_3]            |
|    LCP:      LCP([base+gen],[base+ext1])=[base_1,...,base_1000]     |
|              因为gen_1!=ext1_1，所以LCP停在位置1000                  |
|    截断:     keep_first(1000)->[base_1,...,base_1000]                |
|              丢弃所有generatedtokens！                               |
|    处理:     push_back(ext1_1),push_back(ext1_2),push_back(ext1_3)  |
|    结果:     [base_1,...,base_1000,ext1_1,ext1_2,ext1_3]             |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [请求3:base+ext1,cache_prompt=false]                               |
|                                                                     |
|    旧状态:   [base_1,...,base_1000,ext1_1,ext1_2,ext1_3]            |
|    新请求:   [base_1,...,base_1000,ext1_1,ext1_2,ext1_3]            |
|    参数:     cache_prompt=false                                      |
|    LCP:      (不计算)                                               |
|    截断:     keep_first(0)->[]                                       |
|              全部清空！                                               |
|    处理:     重新处理整个prompt                                      |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [关键结论]                                                         |
|    1.slot.prompt.tokens不会无限增长                                  |
|    2.每次新请求会截断为只保留LCP部分                                 |
|    3.cache_prompt参数控制截断位置                                    |
|       -true:保留LCP                                                 |
|       -false:清空所有tokens                                         |
|                                                                     |
+---------------------------------------------------------------------+
```

### 4.4 生命周期流程图

```
+---------------------------------------------------------------------+
|                    slot.prompt.tokens生命周期流程                     |
+---------------------------------------------------------------------+
|                                                                     |
|     新请求到达                                                      |
|         |                                                          |
|         v                                                          |
|   +------------------+                                              |
|   | 计算LCP          | ----------------------------------+         |
|   | get_common_prefix()                               |         |
|   +------------------+                               |         |
|         |                                          |         |
|         v                                          |         |
|   +------------------+                              |         |
|   | cache_prompt=    |                              |         |
|   | true?            |                              |         |
|   +------------------+                              |         |
|         |                                          |         |
|     Yes-+-> 截断LCP长度                            |         |
|         |                                          |         |
|         v                                          |         |
|   +------------------+                              |         |
|   | cachereuse       | <---------------------------+         |
|   | 处理(可选)        |                                         |
|   +------------------+                              |         |
|         |                                          |         |
|         v                                          |         |
|   +------------------+                              |         |
|   | keep_first       |                              |         |
|   | (截断)           |                              |         |
|   +------------------+                              |         |
|         |                                          |         |
|         +->处理新tokens                                    |
|                   |                                          |
|                   v                                          |
|           +------------------+                               |
|           | push_back()      |                               |
|           | 添加新token       |                               |
|           +------------------+                               |
|                   |                                          |
|                   v                                          |
|           +------------------+                               |
|           | 请求完成          |                               |
|           | slot释放         |                               |
|           +------------------+                               |
|                   |                                          |
|                   v                                          |
|           +------------------+                               |
|           | 保存到Global     |                               |
|           | Cache(可选)      |                               |
|           +------------------+                               |
|                                                                     |
+---------------------------------------------------------------------+
```

## 5. LCP最长公共前缀计算

### 5.1 LCP算法

```cpp
//server-common.cpp:382-430
size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx=std::min(tokens.size(),b.tokens.size());

    if(!has_mtmd){
        //纯文本场景:简单逐个比较
        for(size_t i=0;i<max_idx;++i){
            if(tokens[i]==b.tokens[i]){
                continue;
            }
            return i;  //找到不匹配的位置，返回索引
        }
        return max_idx;  //完全匹配
    }

    //多模态场景:特殊处理LLAMA_TOKEN_NULL
    for(size_t i=0;i<max_idx;++i){
        const llama_token ai=tokens[i];
        const llama_token bi=b.tokens[i];

        if(ai==LLAMA_TOKEN_NULL&&bi==LLAMA_TOKEN_NULL){
            //比较mediachunks的ID和token数量
            const auto & a_chunk=find_chunk(i);
            const auto & b_chunk=b.find_chunk(i);
            const std::string id_ai=mtmd_input_chunk_get_id(a_chunk.get());
            const std::string id_bi=mtmd_input_chunk_get_id(b_chunk.get());
            const size_t n_tok_a=mtmd_input_chunk_get_n_tokens(a_chunk.get());
            const size_t n_tok_b=mtmd_input_chunk_get_n_tokens(b_chunk.get());

            if(id_ai==id_bi&&n_tok_a==n_tok_b){
                i+=n_tok_a-1;  //跳过整个chunk
                continue;
            }
            return i;
        }

        if(ai==bi){
            continue;
        }
        return i;
    }
    return max_idx;
}
```

### 5.2 LCP计算图解

```
+---------------------------------------------------------------------+
|                          LCP计算示例                                  |
+---------------------------------------------------------------------+
|                                                                     |
|  场景1:纯文本                                                        |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  slot.prompt.tokens:  [A B C D E F G]                         | |
|  |                              |                                | |
|  |                              v                                | |
|  |  input_tokens:         [A B C D E F G H I J]                  | |
|  |                                    ↑                          | |
|  |                              LCP=7(ABCDEFG)                   | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
|  场景2:有差异的位置                                                   |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  slot.prompt.tokens:  [A B C D E F G]                         | |
|  |                              |                                | |
|  |                              v                                | |
|  |  input_tokens:         [A B C X E F G H I]                    | |
|  |                             ↑                                 | |
|  |                        LCP=3(ABC)                              | |
|  |                        位置3:D!=X                              | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
|  场景3:多模态内容                                                     |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  slot.prompt.tokens:  [A B NULL NULL NULL D E]                | |
|  |                        (NULL=image_001)                        | |
|  |                              |                                | |
|  |                              v                                | |
|  |  input_tokens:         [A B NULL NULL NULL D E F]             | |
|  |                                    ↑                          | |
|  |                              LCP=7(全部匹配)                   | |
|  |                              比较imageID和token数量            | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
+---------------------------------------------------------------------+
```

### 5.3 LCP计算示例

```
场景:请求2使用base+ext1，请求3使用base+ext1+ext2

请求1完成后:
slot.prompt.tokens=[base_1,base_2,...,base_1000,gen_1]

请求2:
input_tokens=[base_1,...,base_1000,ext1_1,ext1_2,ext1_3]

LCP计算:
  比较位置0-1000:base_i==base_i → 匹配
  比较位置1001:gen_1!=ext1_1 → 不匹配
LCP=1000

请求3:
input_tokens=[base_1,...,base_1000,ext1_1,ext1_2,ext1_3,ext2_1,ext2_2]

LCP计算:
  比较位置0-1003:全部匹配(ext1也匹配)
LCP=1003
```

## 6. 第一层:Slot-levelCacheReuse

### 6.1 工作原理

CacheReuse通过**KVCache位置偏移**来复用之前计算的KV向量。

```cpp
//server-context.cpp:2141-2188
if(can_cache_reuse&&n_cache_reuse>0){
    size_t head_c=n_past;  //cachehead(旧token位置)
    size_t head_p=n_past;  //prompthead(新token位置)

    while(head_c<slot.prompt.tokens.size()&&
          head_p<input_tokens.size()){

        //查找连续匹配的token序列
        size_t n_match=0;
        while(head_c+n_match<slot.prompt.tokens.size()&&
              head_p+n_match<input_tokens.size()&&
              slot.prompt.tokens[head_c+n_match]==input_tokens[head_p+n_match]){
            n_match++;
        }

        if(n_match>=n_cache_reuse){  //阈值检查
            //计算偏移量
            const int64_t kv_shift=(int64_t)head_p-(int64_t)head_c;

            //KVcacheshifting
            llama_memory_seq_rm(llama_get_memory(ctx),slot.id,head_p,head_c);
            llama_memory_seq_add(llama_get_memory(ctx),slot.id,head_c,head_c+n_match,kv_shift);

            //更新n_past
            for(size_t i=0;i<n_match;i++){
                n_past++;  //这些tokens被标记为"已处理"
            }

            head_c+=n_match;
            head_p+=n_match;
        }else{
            head_c+=1;  //跳过不匹配的位置，继续搜索
        }
    }
}
```

### 6.2 CacheReuse图解

```
+---------------------------------------------------------------------+
|                          CacheReuse机制图解                           |
+---------------------------------------------------------------------+
|                                                                     |
|  [初始状态]                                                         |
|                                                                     |
|  slot.prompt.tokens(旧缓存):                                         |
|  +-----+-----+-----+-----+-----+-----+-----+-----+                  |
|  |  A  |  B  |  C  |  D  |  E  |  F  |  G  |  H  |  <-KVCache已计算  |
|  +-----+-----+-----+-----+-----+-----+-----+-----+                  |
|    0     1     2     3     4     5     6     7                       |
|                                                                     |
|  input_tokens(新请求):                                               |
|  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+      |
|  |  A  |  B  |  C  |  D  |  E  |  F  |  G  |  H  |  I  |  J  |      |
|  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+      |
|    0     1     2     3     4     5     6     7     8     9           |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [步骤1:计算LCP]                                                    |
|                                                                     |
|  LCP=8(ABCDEFGH全部匹配)                                            |
|  n_past=8                                                            |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [步骤2:CacheReuse搜索]                                             |
|                                                                     |
|  从head_c=8,head_p=8开始搜索:                                         |
|                                                                     |
|  旧缓存[8]:不存在(size=8)                                            |
|  搜索失败！无法找到>=32的连续chunk                                   |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [正确场景:有足够长的增量]                                            |
|                                                                     |
|  slot.prompt.tokens(旧缓存):                                         |
|  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+    |
|  |  A  |  B  |  C  | ... |  X  |  Y  |  Z  |  X  |  Y  |  Z  |  X  |  Y  |    |
|  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+    |
|    0            ...   10    11    12    13    14    15    16    17    18         |
|                                                                     |
|  input_tokens(新请求):                                               |
|  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+    |
|  |  A  |  B  |  C  | ... |  X  |  Y  |  Z  |  X  |  Y  |  Z  |  X  |  Y  |  Z  |  P  |  Q  |    |
|  +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+    |
|    0            ...   10    11    12    13    14    15    16    17    18    19    20    21         |
|                                                                                  ↑           |
|                                                                          新增部分:PQ         |
|                                                                     |
|  LCP=18                                                               |
|  从位置18开始搜索:                                                    |
|    input_tokens[18:]=XYZPQ                                          |
|    旧缓存[18:]=不存在                                                 |
|    无法重用                                                          |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  [实际有效场景:cache_reuse重用之前的增量]                              |
|                                                                     |
|  请求1完成后:                                                        |
|  slot.prompt.tokens=[base_1,...,base_1000,inc_1,inc_2,...,inc_40]    |
|                      ↑                                              |
|                      40个连续增量tokens                               |
|                                                                     |
|  请求2:                                                              |
|  input_tokens=[base_1,...,base_1000,inc_1,...,inc_40,new_1,new_2]    |
|                                                                     |
|  LCP=1040                                                            |
|  从位置1040开始搜索:                                                  |
|    input_tokens[1040:]=new_1,new_2                                    |
|    slot.prompt.tokens[1040:]=空(只有1040个tokens)                    |
|    仍然无法重用！                                                     |
|                                                                     |
|  结论:cache_reuse主要用于**跨请求重用相同的新增部分**                   |
|                                                                     |
+---------------------------------------------------------------------+
```

### 6.3 cache_reuse适用场景

```
+---------------------------------------------------------------------+
|                      cache_reuse适用场景                              |
+---------------------------------------------------------------------+
|                                                                     |
|  场景A:精确重复请求(缓存最佳)                                         |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  请求1:prompt=[A B C D E]                                      | |
|  |         缓存区=[A B C D E]                                     | |
|  |                                                                | |
|  |  请求2:prompt=[A B C D E]                                      | |
|  |         LCP=5                                                  | |
|  |         n_past=5                                               | |
|  |         全部复用！TTFT大幅降低                                  | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
|  场景B:前缀匹配+新内容(部分复用)                                       |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  请求1:prompt=[A B C D E]                                      | |
|  |         缓存区=[A B C D E]                                     | |
|  |                                                                | |
|  |  请求2:prompt=[A B C D E F G H I J K L M N O P Q R S T U V W X]| |
|  |         LCP=5                                                  | |
|  |         新增部分>=32tokens                                      | |
|  |         cache_reuse无法复用(搜索从位置5开始，但旧缓存只有5个)    | |
|  |         但新增部分>=32tokens可以在下次请求时复用                 | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
|  场景C:增量太短(无法复用)                                            |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  请求1:prompt=[A B C D E]                                      | |
|  |         缓存区=[A B C D E]                                     | |
|  |                                                                | |
|  |  请求2:prompt=[A B C D E F G H I]  <-新增3个tokens             | |
|  |         新增<32tokens                                          | |
|  |         cache_reuse不工作                                      | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
+---------------------------------------------------------------------+
```

### 6.4 cache_reuse与LCP的关系

```
cache_reuse的工作原理:

1.LCP部分:已经在KVCache中，不需要重新计算
   -n_past=LCP长度
   -从位置n_past开始处理新tokens

2.cache_reuse:复用LCP之后、>=n_cache_reuse的连续匹配chunk
   -搜索范围:从n_past开始
   -复用条件:连续匹配tokens>=n_cache_reuse
   -效果:跳过这些tokens的KV计算

3.新tokens:需要重新计算KV

        |----LCP----|---cache_reuse---|----新tokens----|
        v           v                 v                 v
    [A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f]
    0           10                25                40
                ↑                 ↑
              n_past           n_past+n_match

注意:cache_reuse搜索从n_past开始，所以它复用的是
    "之前请求处理过、当前请求也有的"增量部分
```

## 7. 第二层:GlobalPromptCache

### 7.1 概述

GlobalPromptCache是**跨slot共享**的完整promptstate缓存，使用LRU淘汰策略。

```cpp
//server-task.cpp:1432-1482
bool server_prompt_cache::load(server_prompt & prompt,const server_tokens & tokens_new,
                               llama_context * ctx,int32_t id_slot){
    //计算当前slot.prompt与新请求的LCP
    const int lcp_best=prompt.tokens.get_common_prefix(tokens_new);
    float f_keep_best=float(lcp_best)/prompt.tokens.size();
    float sim_best=float(lcp_best)/tokens_new.size();

    //在全局缓存中搜索最相似的prompt
    for(auto it=states.begin();it!=states.end();++it){
        const int lcp_cur=it->tokens.get_common_prefix(tokens_new);
        float f_keep_cur=float(lcp_cur)/it->tokens.size();
        float sim_cur=float(lcp_cur)/tokens_new.size();

        //选择最佳匹配(同时满足f_keep和sim更高)
        if(f_keep_best<f_keep_cur&&sim_best<sim_cur){
            f_keep_best=f_keep_cur;
            sim_best=sim_cur;
            it_best=it;
        }
    }

    if(it_best!=states.end()){
        //恢复KVcachestate
        const size_t size=it_best->data.size();
        const size_t n=llama_state_seq_set_data_ext(ctx,it_best->data.data(),size,id_slot,0);

        if(n!=size){
            SLT_WRN(slot,"failed to restore state with size%zu\n",size);
            return false;
        }

        //释放memory
        it_best->data.clear();
        it_best->data.shrink_to_fit();

        //关键:完全替换slot.prompt！使用移动语义
        prompt=std::move(*it_best);

        //从缓存中移除(已被使用)
        states.erase(it_best);
    }

    return true;
}
```

### 7.2 GlobalCache图解

```
+---------------------------------------------------------------------+
|                      GlobalPromptCache工作流程                        |
+---------------------------------------------------------------------+
|                                                                     |
|  [步骤1:请求到达，选择slot0处理]                                      |
|                                                                     |
|    Slot0:prompt=[A B C D E]  <-当前状态                             |
|    GlobalCache:                                                      |
|      +-----------------------------------------------------------+  |
|      | state0:[A B C D E F G H I]  (size=200MiB)                |  |
|      | state1:[X Y Z]  (size=50MiB)                            |  |
|      | state2:[Hello World...]  (size=100MiB)                  |  |
|      +-----------------------------------------------------------+  |
|                                                                     |
|  [步骤2:查找GlobalCache]                                            |
|                                                                     |
|    新请求tokens:[A B C D E F G H I J K]                              |
|                                                                     |
|    搜索所有state:                                                    |
|    -state0:LCP([A..I],[A..K])=9,sim=9/11=82%,f_keep=9/9=100%        |
|    -state1:LCP([X..Z],[A..K])=0                                     |
|    -state2:LCP=0                                                     |
|                                                                     |
|    最佳匹配:state0(sim=82%,f_keep=100%)                              |
|                                                                     |
|  [步骤3:完全替换slot.prompt]                                         |
|                                                                     |
|    slot0.prompt=[A B C D E]  <-旧状态                               |
|                   ↓                                                 |
|                   ↓ std::move(*state0)                              |
|                   ↓                                                 |
|    slot0.prompt=[A B C D E F G H I]  <-新状态(state0内容)           |
|                                                                     |
|    同时:                                                             |
|    -恢复state0的KVcachestate到slot0                                  |
|    -从GlobalCache中移除state0                                        |
|    -slot0可以直接使用已有的KVCache                                   |
|                                                                     |
|  [步骤4:slot释放时保存到GlobalCache]                                 |
|                                                                     |
|    slot0.prompt=[A B C D E F G H I]                                 |
|                          ↓                                          |
|                          ↓ prompt_save()                            |
|                          ↓                                          |
|    GlobalCache:                                                      |
|      +-----------------------------------------------------------+  |
|      | state1:[X Y Z]                                            |  |
|      | state2:[Hello World...]                                  |  |
|      | state0:[A B C D E F G H I]  <-重新插入，成为最新          |  |
|      +-----------------------------------------------------------+  |
|                                                                     |
+---------------------------------------------------------------------+
```

### 7.3 LRU淘汰策略

```cpp
//server-task.cpp:1484-1524
void server_prompt_cache::update(){
    //基于size的淘汰
    if(limit_size>0){
        //始终保留至少一个state
        while(states.size()>1&&size()>limit_size){
            SRV_WRN(" - cache size limit reached, removing oldest entry\n");
            states.pop_front();  //移除最老的entry(LRU)
        }
    }

    //动态计算token限制
    const float size_per_token=std::max<float>(1.0f,float(size())/std::max<size_t>(1,n_tokens()));
    const size_t limit_tokens_cur=limit_size>0?
        std::max<size_t>(limit_tokens,limit_size/size_per_token):limit_tokens;

    //基于token数量的淘汰
    if(limit_tokens>0){
        while(states.size()>1&&n_tokens()>limit_tokens_cur){
            SRV_WRN(" - cache token limit reached, removing oldest entry\n");
            states.pop_front();  //LRU淘汰
        }
    }
}
```

```
+---------------------------------------------------------------------+
|                         LRU淘汰策略图解                                |
+---------------------------------------------------------------------+
|                                                                     |
|  初始状态(容量8192MiB):                                              |
|                                                                     |
|    GlobalCache:                                                      |
|    +-----------------------------------------------------------+    |
|    | state0:[A B C D E]  200MiB  ←最老                           |    |
|    | state1:[F G H I J]  200MiB                                 |    |
|    | state2:[K L M N O]  200MiB                                 |    |
|    | state3:[P Q R S T]  200MiB  ←最新                           |    |
|    +-----------------------------------------------------------+    |
|    Total:800MiB                                                      |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  新请求处理完成，保存到Cache:                                         |
|                                                                     |
|    新state:[U V W X Y]  200MiB                                       |
|                                                                     |
|    GlobalCache:                                                      |
|    +-----------------------------------------------------------+    |
|    | state0:[A B C D E]  200MiB                                 |    |
|    | state1:[F G H I J]  200MiB                                 |    |
|    | state2:[K L M N O]  200MiB                                 |    |
|    | state3:[P Q R S T]  200MiB                                 |    |
|    | state4:[U V W X Y]  200MiB  ←最新                           |    |
|    +-----------------------------------------------------------+    |
|    Total:1000MiB                                                     |
|                                                                     |
|  ================================================================   |
|                                                                     |
|  容量达到限制，触发LRU淘汰:                                           |
|                                                                     |
|    pop_front() -> 移除state0                                         |
|                                                                     |
|    GlobalCache:                                                      |
|    +-----------------------------------------------------------+    |
|    | state1:[F G H I J]  200MiB  ←最老                           |    |
|    | state2:[K L M N O]  200MiB                                 |    |
|    | state3:[P Q R S T]  200MiB                                 |    |
|    | state4:[U V W X Y]  200MiB  ←最新                           |    |
|    +-----------------------------------------------------------+    |
|    Total:800MiB                                                      |
|                                                                     |
+---------------------------------------------------------------------+
```

### 7.4 GlobalCache与slot-levelcache的区别

```
+---------------------------------------------------------------------+
|                    GlobalCache vs Slot-levelCache                     |
+---------------------------------------------------------------------+
|                                                                     |
|                    GlobalPromptCache      Slot-levelCacheReuse       |
|  ┌──────────────────────────────────────────────────────────────┐   │
|  |                                                              |   │
|  |  作用范围       跨所有slots              同一slot              |   │
|  |                                                              |   │
|  |  存储内容       完整promptstate          部分KVCache          |   │
|  |                -tokens                  -KV向量              |   │
|  |                -KVcachestate            -位置偏移            |   |
|  |                -checkpoints                                    |   │
|  |                                                              |   |
|  |  命中条件       LCP相似度>=25%           连续匹配>=32tokens    |   │
|  |              f_keep>=0.25                                       |   |
|  |              sim同时更高                                        |   |
|  |                                                              |   |
|  |  效果           完全替换slot.prompt      跳过部分KV计算        |   │
|  |              恢复完整state                                      |   |
|  |                                                              |   |
|  |  容量限制       8192MiB+token数          无                   |   │
|  |              LRU淘汰                                           |   |
|  |                                                              |   │
|  |  Multi-modal    支持                    不支持                |   |
|  |                                                              |   |
|  └──────────────────────────────────────────────────────────────┘   |
|                                                                     |
|  互补关系:                                                           |
|  1.GlobalCache:跨请求(不同slot)、完全不同的请求                       |
|  2.Slot-level:同一slot的连续请求                                      |
|  3.两者可以同时工作                                                  |
|                                                                     |
+---------------------------------------------------------------------+
```

## 8. cache_prompt参数

### 8.1 参数作用

`cache_prompt`参数控制是否启用promptcaching机制。

```cpp
//server-context.cpp:2120-2191

if(slot.task->params.cache_prompt){
    //cache_prompt=true:启用promptcaching
    n_past=slot.prompt.tokens.get_common_prefix(input_tokens);
    //...cache_reuse处理...
}else{
    //cache_prompt=false:禁用promptcaching
    n_past=0;
}

//无论cache_prompt值如何，keep_first都会被调用
slot.prompt.tokens.keep_first(n_past);
```

### 8.2 cache_prompt=truevs=false对比

```
+---------------------------------------------------------------------+
|                    cache_prompt参数对比                               |
+---------------------------------------------------------------------+
|                                                                     |
|  cache_prompt=true(默认)                                            |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  请求1:                                                        | |
|  |    slot.prompt=[]                                             | |
|  |    input=[A B C D E]                                          | |
|  |    LCP=0                                                      | |
|  |    keep_first(0)->[]                                          | |
|  |    处理整个prompt                                             | |
|  |    slot.prompt=[A B C D E]                                    | |
|  |                                                                | |
|  |  请求2(相同prompt):                                           | |
|  |    slot.prompt=[A B C D E]                                    | |
|  |    input=[A B C D E]                                          | |
|  |    LCP=5                                                      | |
|  |    keep_first(5)->[A B C D E]                                 | |
|  |    n_past=5                                                   | |
|  |    跳过KV计算，直接生成                                        | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
|  cache_prompt=false                                                 |
|  +----------------------------------------------------------------+ |
|  |                                                                | |
|  |  请求1:                                                        | |
|  |    slot.prompt=[]                                             | |
|  |    input=[A B C D E]                                          | |
|  |    cache_prompt=false                                         | |
|  |    n_past=0                                                   | |
|  |    keep_first(0)->[]                                          | |
|  |    处理整个prompt                                             | |
|  |    slot.prompt=[A B C D E]                                    | |
|  |                                                                | |
|  |  请求2(相同prompt):                                           | |
|  |    slot.prompt=[A B C D E]                                    | |
|  |    input=[A B C D E]                                          | |
|  |    cache_prompt=false                                         | |
|  |    n_past=0                                                   | |
|  |    keep_first(0)->[]                                          | |
|  |    重新处理整个prompt！                                        | |
|  |                                                                | |
|  +----------------------------------------------------------------+ |
|                                                                     |
|  结论:cache_prompt=false会强制重新处理整个prompt                      |
|                                                                     |
+---------------------------------------------------------------------+
```

### 8.3 API使用方式

```json
//通过API请求启用cache_prompt
POST/v1/chat/completions
{
    "model":"llama3.1-8b",
    "messages":[{"role":"user","content":"你的问题"}],
    "cache_prompt":true  //启用promptcaching(默认true)
}
```

```bash
#通过命令行参数
--cache-prompt  #启用(默认)
--no-cache-prompt  #禁用
```

### 8.4 使用场景

```
+---------------------------------------------------------------------+
|                    cache_prompt使用场景                               |
+---------------------------------------------------------------------+
|                                                                     |
|  cache_prompt=true(推荐):                                           |
|  1.连续对话，同一用户多次请求                                         |
|  2.需要复用之前计算的KVCache                                         |
|  3.对话历史累积场景                                                  |
|                                                                     |
|  cache_prompt=false:                                                |
|  1.每次请求都是独立的prompt                                          |
|  2.调试或测试场景                                                    |
|  3.需要确保安全隔离的场景                                             |
|  4.嵌入式模型输出(embdding)场景                                      |
|                                                                     |
+---------------------------------------------------------------------+
```

## 9. 多模态场景处理

### 9.1 Multi-modal限制

```cpp
//server-context.cpp:2132-2134
const bool can_cache_reuse=
    llama_memory_can_shift(llama_get_memory(ctx))&&
    !slot.prompt.tokens.has_mtmd;  //多模态时禁用cache_reuse

if(!can_cache_reuse&&n_cache_reuse>0){
    SLT_WRN(slot,"cache reuse is not supported - ignoring n_cache_reuse=%d\n",n_cache_reuse);
}
```

### 9.2 多模态token处理

```
+---------------------------------------------------------------------+
|                      多模态token处理                                  |
+---------------------------------------------------------------------+
|                                                                     |
|  多模态prompt示例:                                                   |
|                                                                     |
|  文本:[Hello][world]                                                |
|  图片:[img_001:3tokens]                                             |
|  文本:[please][describe][this]                                      |
|                                                                     |
|  tokens数组:                                                        |
|  +-----+-----+---------+---------+-----+-----+-----+                |
|  |  H  |  w  |  NULL   |  NULL   | p  |  l  | ... |                |
|  +-----+-----+---------+---------+-----+-----+-----+                |
|    0     1      2         3        4     5     6                    |
|                 ↑                                    |              |
|                 +--LLAMA_TOKEN_NULL------------------+              |
|                      (image_001占位符)                              |
|                                                                     |
|  map_idx_to_media:                                                  |
|  +-----------------------------------------------------------+     |
|  |  index2->ImageChunk(id="img_001",n_tokens=3)             |     |
|  +-----------------------------------------------------------+     |
|                                                                     |
+---------------------------------------------------------------------+
```

### 9.3 多模态LCP计算

```cpp
//server-common.cpp:397-429
for(size_t i=0;i<max_idx;++i){
    const llama_token ai=tokens[i];
    const llama_token bi=b.tokens[i];

    if(ai==LLAMA_TOKEN_NULL&&bi==LLAMA_TOKEN_NULL){
        //比较mediachunks
        const auto & a_chunk=find_chunk(i);
        const auto & b_chunk=b.find_chunk(i);

        const std::string id_ai=mtmd_input_chunk_get_id(a_chunk.get());
        const std::string id_bi=mtmd_input_chunk_get_id(b_chunk.get());

        const size_t n_tok_a=mtmd_input_chunk_get_n_tokens(a_chunk.get());
        const size_t n_tok_b=mtmd_input_chunk_get_n_tokens(b_chunk.get());

        //只有当ID和token数量都相同时才匹配
        if(id_ai==id_bi&&n_tok_a==n_tok_b){
            i+=n_tok_a-1;  //跳过整个chunk
            continue;
        }
        return i;  //不匹配
    }

    if(ai==bi){
        continue;
    }
    return i;
}
```

### 9.4 多模态keep_first保护

```cpp
//server-common.cpp:344-356
//不能在image中间截断
if(n>0){
    if(tokens[n-1]==LLAMA_TOKEN_NULL&&tokens[n]==LLAMA_TOKEN_NULL){
        find_chunk(n-1);  //抛出错误(不是chunk起始位置)
    }
}

//示例:
//tokens:[0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
//          1   2   3   4   5    6      7      8      9     10
//允许截断:                          ^                    ^
//禁止截断:                                ^      ^             ^
//              (在image中间截断会抛出错误)
```

```
+---------------------------------------------------------------------+
|                    多模态截断保护规则                                  |
+---------------------------------------------------------------------+
|                                                                     |
|  原始tokens:                                                         |
|  [T1][T2][T3][T4][T5][NULL][NULL][NULL][NULL]                       |
|   1   2   3   4   5    6      7      8      9                       |
|                      ↑                                              |
|                      img_chunk占用4个位置                            |
|                                                                     |
|  允许的截断位置:                                                      |
|  [T1][T2][T3][T4][T5]              ←截断位置5                       |
|   1   2   3   4   5                                                      |
|                      ↑                                              |
|                      边界:完整保留image                              |
|                                                                     |
|  [T1][T2][T3][T4][T5][NULL][NULL][NULL][NULL]  ←不截断              |
|   1   2   3   4   5    6      7      8      9                       |
|                                                                     |
|  禁止的截断位置:                                                      |
|  [T1][T2][T3][T4]        ←截断位置4                                  |
|   1   2   3   4                         ↑                           |
|                      NULL在位置4和5都是NULL                          |
|                      错误:在image中间截断！                          |
|                                                                     |
|  [T1][T2][T3][T4][T5][NULL]      ←截断位置6                         |
|   1   2   3   4   5    6                         ↑                  |
|                      NULL在位置5是，位置6也是                         |
|                      错误:在image中间截断！                          |
|                                                                     |
+---------------------------------------------------------------------+
```

## 10. 测试数据分析

### 10.1 测试结果

| 测试类型 | Round | TTFT | 比率 | 预期 | 实际 | 说明 |
|---------|-------|------|------|------|------|------|
| 精确匹配 | R1 | 2225ms | - | baseline | OK | - |
| 精确匹配 | R2 | 42ms | 1.91% | <50% | OK | cache_reuse工作 |
| 增长前缀 | R1 | 2106ms | - | baseline | OK | - |
| 增长前缀 | R2 | 2290ms | 108% | <30% | FAIL | ext1太短，cache_reuse不工作 |
| 增长前缀 | R3 | 256ms | 12% | <30% | OK | Globalpromptcache命中 |
| 增长前缀 | R4 | 2101ms | 99% | <30% | FAIL | Globalcache未命中 |

### 10.2 精确匹配请求分析

```
+---------------------------------------------------------------------+
|                      精确匹配请求时序图                                |
+---------------------------------------------------------------------+
|                                                                     |
|  请求1:                                                             |
|  +--------------------------------------------------------------+   |
|  | input:[base_1,...,base_1000]                                |   |
|  | slot.prompt:[]                                              |   |
|  | LCP=0                                                       |   |
|  | 处理:计算1000个tokens的KVCache                               |   |
|  | TTFT:2225ms                                                 |   |
|  | 释放:slot.prompt=[base_1,...,base_1000]                     |   |
|  | 保存:GlobalCache新增state0=[base_1,...,base_1000]           |   |
|  +--------------------------------------------------------------+   |
|                           |                                         |
|                           v                                         |
|  请求2:                                                             |
|  +--------------------------------------------------------------+   |
|  | input:[base_1,...,base_1000]                                |   |
|  | slot.prompt:[base_1,...,base_1000]  <-来自请求1             |   |
|  | LCP=1000                                                    |   |
|  | 处理:跳过所有KV计算，直接生成                                |   |
|  | TTFT:42ms(98%加速!)                                         |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+
```

### 10.3 增长前缀请求分析

```
+---------------------------------------------------------------------+
|                      增长前缀请求时序图                                |
+---------------------------------------------------------------------+
|                                                                     |
|  请求1:                                                             |
|  +--------------------------------------------------------------+   |
|  | input:[base_1,...,base_1000]                                |   |
|  | LCP=0                                                       |   |
|  | 处理:计算1000个tokens                                        |   |
|  | TTFT:2106ms                                                 |   |
|  | slot.prompt=[base_1,...,base_1000]                          |   |
|  +--------------------------------------------------------------+   |
|                           |                                         |
|                           v                                         |
|  请求2:base+ext1(ext1=3tokens)                                      |
|  +--------------------------------------------------------------+   |
|  | input:[base_1,...,base_1000,ext1_1,ext1_2,ext1_3]           |   |
|  | slot.prompt:[base_1,...,base_1000]                          |   |
|  | LCP=1000                                                    |   |
|  | cache_reuse:从位置1000开始搜索，旧缓存只有1000个tokens        |   |
|  |             搜索失败(ext1只有3tokens<32)                     |   |
|  | 处理:重新计算ext1的KV(3tokens)                               |   |
|  | TTFT:2290ms(108%比请求1慢!)                                  |   |
|  +--------------------------------------------------------------+   |
|                           |                                         |
|                           v                                         |
|  请求3:base+ext1+ext2(ext2=3tokens)                                 |
|  +--------------------------------------------------------------+   |
|  | input:[base_1,...,base_1000,ext1_1,ext1_2,ext1_3,ext2_1,ext2_2,ext2_3]|   |
|  | LCP=1003                                                    |   |
|  | GlobalCache查找:命中请求1的state0                            |   |
|  |             完全替换slot.prompt=[base_1,...,base_1000]      |   |
|  |             恢复KVCache                                      |   |
|  | TTFT:256ms(12%)                                             |   |
|  +--------------------------------------------------------------+   |
|                           |                                         |
|                           v                                         |
|  请求4:base+ext1+ext2+ext3                                          |
|  +--------------------------------------------------------------+   |
|  | input:更长                                                  |   |
|  | GlobalCache:请求3的state可能已被LRU淘汰                      |   |
|  |             未找到匹配                                        |   |
|  | TTFT:2101ms(99%)                                            |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
+---------------------------------------------------------------------+
```

### 10.4 增长前缀失败原因

```
增长前缀测试失败的根本原因:

1.增量太短
   ext1/ext2/ext3只有3-5tokens，远小于n_cache_reuse=32

2.cache_reuse搜索范围限制
   cache_reuse只搜索LCP之后的增量部分
   增量<32，无法触发复用

3.GlobalCache的不确定性
   命中取决于:
   -之前是否有其他slot处理过相同的prompt
   -缓存容量限制(8192MiB)
   -LRU淘汰策略
   -相似度阈值(f_keep>=0.25)

正确做法:确保每次增量>=32tokens
```

## 11. 常见问题与解决方案

### 11.1 Q&A

```
Q:为什么slot.prompt.tokens不会无限增长？

A:因为每次新请求到达时，会调用keep_first(n_past)截断tokens。
  只保留与新请求的LCP部分，非LCP部分被丢弃。


Q:cache_reuse和GlobalCache有什么区别？

A:
  -cache_reuse:同一slot内，复用之前请求的KVCache位置
  -GlobalCache:跨slot共享，存储完整promptstate
  -两者可以同时工作，互为补充


Q:为什么增长前缀测试中cache_reuse不工作？

A:因为ext1只有3tokens，而n_cache_reuse=32。
  cache_reuse要求连续匹配>=32tokens才能触发。
  增量太短，无法满足阈值。


Q:GlobalCache一定会命中吗？

A:不一定。GlobalCache使用LRU淘汰策略，可能被淘汰。
  而且命中需要满足相似度条件(f_keep>=25%)。
  增长前缀测试中Round3命中可能是跨slot的缓存命中。


Q:cache_prompt=false会怎样？

A:cache_prompt=false时:
  -n_past=0
  -keep_first(0)清空所有tokens
  -每次请求都重新处理整个prompt
  -禁用promptcaching功能
```

### 11.2 调试技巧

```
1.启用调试日志
  --slots-debug  #启用slot调试输出

2.查看KVCache状态
  在日志中搜索:
  -"prompt_tokens"  #查看prompttokens
  -"reusing chunk"  #cache_reuse命中
  -"updating prompt cache"  #保存到GlobalCache

3.监控缓存命中率
  -观察TTFT变化
  -比较不同请求的TTFT比率
```

## 12. 最佳实践

### 12.1 优化建议

```
+---------------------------------------------------------------------+
|                    PrefixCache优化建议                                |
+---------------------------------------------------------------------+
|                                                                     |
|  1.保持prompt前缀稳定                                                |
|     -对话系统中，每个请求包含完整对话历史                             |
|     -避免只发送增量内容                                               |
|                                                                     |
|  2.确保增量足够长                                                    |
|     -每次新增内容>=32tokens                                          |
|     -或者使用--cache-reuse=1降低阈值                                 |
|                                                                     |
|  3.使用固定的slot                                                    |
|     通过id_slot参数强制使用同一slot                                  |
|     提高slot-levelcache_reuse的命中率                                |
|                                                                     |
|  4.调整GlobalCache大小                                               |
|     --cache-prompt-size 16384  #增加到16GiB                          |
|     减少LRU淘汰的概率                                                |
|                                                                     |
|  5.启用cache_prompt(默认)                                           |
|     除非有特殊需求，否则不要禁用                                      |
|                                                                     |
+---------------------------------------------------------------------+
```

### 12.2 客户端实现建议

```json
//推荐:每次请求发送完整对话历史
{
    "messages":[
        {"role":"system","content":"你是AI助手"},
        {"role":"user","content":"写一篇关于AI的文章"},
        {"role":"assistant","content":"以下是文章..."},
        {"role":"user","content":"继续"}  //包含完整历史
    ],
    "cache_prompt":true
}
```

```python
#客户端维护对话历史的示例
class ConversationManager:
    def __init__(self):
        self.history=[]
    
    def add_user_message(self,content):
        self.history.append({"role":"user","content":content})
    
    def add_assistant_message(self,content):
        self.history.append({"role":"assistant","content":content})
    
    def get_full_messages(self):
        #返回完整历史，每次请求发送
        return self.history.copy()
    
    def clear(self):
        self.history=[]
```

### 12.3 服务器配置

```bash
#推荐配置
./llama-server \
    -m model.gguf \
    --port 8080 \
    -c 8192 \
    --cache-reuse 32 \         #默认阈值
    --cache-prompt-size 8192 \  #默认容量
    --cache-prompt-tokens 0 \   #无token数量限制
    --n-slots 4                 #4个slot
```

### 12.4 性能优化对比

```
+---------------------------------------------------------------------+
|                    优化效果对比                                       |
+---------------------------------------------------------------------+
|                                                                     |
|  场景:连续对话，每次新增内容                                          |
|                                                                     |
|  优化前(不发送完整历史):                                             |
|  +--------------------------------------------------------------+   |
|  | Round1:TTFT=2000ms  处理base                                 |   |
|  | Round2:TTFT=2100ms  处理ext1(3tokens) cache_reuse不工作      |   |
|  | Round3:TTFT=2200ms  处理ext2(3tokens)                        |   |
|  | Round4:TTFT=2150ms  处理ext3(3tokens)                        |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
|  优化后(发送完整历史):                                               |
|  +--------------------------------------------------------------+   |
|  | Round1:TTFT=2000ms  处理完整历史                             |   |
|  | Round2:TTFT=50ms    复用所有历史cache                        |   |
|  | Round3:TTFT=55ms    复用更新后的历史cache                    |   |
|  | Round4:TTFT=48ms    复用更新后的历史cache                    |   |
|  +--------------------------------------------------------------+   |
|                                                                     |
|  加速比:~40x                                                        |
|                                                                     |
+---------------------------------------------------------------------+
```

## 13. 关键代码位置

### 13.1 核心代码文件

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| **cache_prompt参数** | server-task.cpp | 50,170 | 定义和默认值 |
| **cache_prompt检查** | server-context.cpp | 2120 | if(slot.task->params.cache_prompt) |
| **Token截断** | server-context.cpp | 2317 | slot.prompt.tokens.keep_first(n_past) |
| **Token添加(text)** | server-context.cpp | 2421 | push_back(cur_tok) |
| **Token添加(media)** | server-context.cpp | 2364 | push_back(chunk.get()) |
| **Token添加(生成)** | server-context.cpp | 1987,2013 | push_back(slot.sampled) |
| **LCP计算(text)** | server-common.cpp | 385-394 | 纯文本比较 |
| **LCP计算(mtmd)** | server-common.cpp | 397-429 | 包含mediachunk比较 |
| **Cachereuse实现** | server-context.cpp | 2141-2188 | KVcacheshifting |
| **Cachereuse mtmd检查** | server-context.cpp | 2133-2134 | !slot.prompt.tokens.has_mtmd |
| **keep_first实现** | server-common.cpp | 338-369 | 截断并清理mediachunks |
| **Image中间截断保护** | server-common.cpp | 350-356 | 检查是否在image中间截断 |
| **Globalcache加载** | server-task.cpp | 1432-1482 | prompt=std::move(*it_best) |
| **Globalcache LRU** | server-task.cpp | 1484-1524 | states.pop_front() |
| **Globalcache调用** | server-context.cpp | 997 | prompt_load(*prompt_cache,task.tokens) |
| **SWA checkpoint恢复** | server-context.cpp | 2258-2283 | 搜索并恢复checkpoint |
| **SWA checkpoint创建** | server-context.cpp | 2468-2491 | 创建新checkpoint |
| **server_tokens结构** | server-common.h | 127-220 | tokens数据结构定义 |
| **slot.release()** | server-context.cpp | 300-314 | 保留prompt.tokens |
| **slot.reset()** | server-context.cpp | 176-204 | 清空task但保留prompt.tokens |
| **clear_slot()** | server-context.cpp | 1010-1019 | 清空prompt.tokens |
| **can_split()** | server-context.cpp | 220-224 | 检查是否可以splitprompt |

### 13.2 配置参数

```cpp
//common/arg.cpp
--cache-reuse N      //默认值:0=disabled,server=32
--cache-prompt-size  //默认:8192MiB
--cache-prompt-tokens  //默认:0=无限制
--n-slots N          //默认:8
```

### 13.3 命令行参数

```bash
#启用cache_reuse
--cache-reuse 32     #连续匹配>=32tokens时复用

#配置GlobalCache
--cache-prompt-size 16384  #16GiB
--cache-prompt-tokens 100000  #10万tokens

#调试
--slots-debug  #启用slot调试输出
```

---

## 总结

### 核心概念

1.**slot.prompt.tokens不会无限增长**
   -每次新请求到达时被截断为LCP长度
   -由keep_first(n_past)实现

2.**两层缓存机制**
   -Slot-levelCacheReuse:同一slot，KVcacheshifting
   -GlobalPromptCache:跨slot，完整state替换

3.**cache_prompt参数**
   -true:启用promptcaching(默认)
   -false:禁用，每次重新处理

4.**cache_reuse阈值**
   -默认32tokens
   -增量<32无法触发

5.**Multi-modal限制**
   -cache_reuse不支持多模态
   -GlobalCache支持多模态
   -keep_first有image中间截断保护

### 性能优化关键

-发送完整对话历史
-确保增量>=32tokens
-使用固定slot
-增大GlobalCache容量
-启用cache_prompt(默认)

---

*文档版本:1.0*
*最后更新:2026-01-28*
*参考:llama.cpp源代码和测试用例*

