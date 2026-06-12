# llama2.cu · 代码导读 & 概念 FAQ

> 配套 `docs/transformer-inference.md`(讲推理流程)。本篇侧重**从入口读代码、核心数据结构、模型文件布局、关键维度概念**,以问答形式整理。
> 示例数值来自实测 `stories15M.bin`:`dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, head_size=48, vocab=32000, seq_len=256`。

---

## 目录

1. [程序入口与阅读顺序](#1-程序入口与阅读顺序)
2. [命令行参数](#2-命令行参数)
3. [run.c vs runq.c](#3-runc-vs-runqc)
4. [核心数据结构](#4-核心数据结构)
4.5 [三个 build 函数:初始化流程](#45-三个-build-函数初始化流程)
5. [模型文件 .bin 布局](#5-模型文件-bin-布局)
6. [关键维度概念](#6-关键维度概念-dim--hidden_dim--head_size--seq_len)
7. [hidden_states / 隐藏状态](#7-hidden_states--隐藏状态)
8. [Tokenizer 分词器](#8-tokenizer-分词器)(含 score 含义 & BPE 合并追踪)
9. [运行时主流程:generate / forward / sample](#9-运行时主流程generate--forward--sample)(含 prefill/decode、lm_head、Sampler)
10. [观测日志](#10-观测日志read_checkpoint--memory_map_weights)

---

## 1. 程序入口与阅读顺序

入口是标准 `main()`(`run.cu:1251`),做 4 件事:

```
main()
├─ 1. 解析命令行         (argv[1]=模型路径;-t -p -s -n -i -z -m -y)
├─ 2. 构建三大对象
│     ├─ build_transformer()  ← mmap 加载权重 + 读 config
│     ├─ build_tokenizer()    ← 加载词表
│     └─ build_sampler()      ← 初始化采样器
├─ 3. create_cublas_handle()  ← GPU 版初始化 cuBLAS
└─ 4. generate() 或 chat()    ← 主循环
```

**建议阅读顺序**:

| 步骤 | 看什么 | 位置 | 重点 |
|------|--------|------|------|
| ① | `main()` | 1251 | 建立全局观 |
| ② | `read_checkpoint` / `memory_map_weights` | 237 / 206 | mmap、读 config、权重切分 |
| ③ | `generate()` | 主循环 | prefill/decode 共用、KV Cache |
| ④ | `forward()` | 623 | ⭐核心:一个 token 过 6 层 |
| ⑤ | 各 kernel | 301–621 | rmsnorm/RoPE/attention/SwiGLU |
| ⑥ | `sample()` | 1036 | logits → next token |
| ⑦ | `encode`/`decode` | ~755/~782 | BPE 分词 |

---

## 2. 命令行参数

```
./runcuda <模型文件> [选项...]
```

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `<模型文件>` | 路径 | 必传 | 第一个位置参数,如 `stories15M.bin` |
| `-t` | float | 1.0 | **温度**:0=贪婪(确定);1.0=原始;越大越随机 |
| `-p` | float | 0.9 | **top-p** 核采样:只在累积概率前 p 的词里采样;1.0=关闭 |
| `-s` | int | 时间 | **随机种子**:固定可复现 |
| `-n` | int | 256 | **生成步数**:0 或超 seq_len 会截到 seq_len |
| `-i` | string | 无 | **输入 prompt** |
| `-z` | string | tokenizer.bin | 自定义分词器路径 |
| `-m` | string | generate | 模式:`generate` / `chat` |
| `-y` | string | 无 | system prompt(仅 chat) |

**调参直觉**:要稳定→`-t` 调低或 `-t 0`;要创意→`-t` 调高;要复现→固定 `-s`。

---

## 3. run.c vs runq.c

| | `run.c` | `runq.c` |
|--|---------|----------|
| 精度 | **fp32** | **int8 量化** |
| 权重类型 | `float*` | `QuantizedTensor`(`int8_t* q` + `float* s`) |
| 模型文件 | v0/v1 格式 | version2 量化导出 |
| 体积 | 基准 | **小 ~4×** |
| 速度 | 基准 | **快 ~3×** |

**为什么大量代码重复**:量化只影响"权重存储 + matmul"这条线,其余(分词、采样、生成循环、main、非 matmul 算子)完全一样,原作者直接复制一份再改关键路径 —— 教学项目刻意不抽象,宁可重复也要让每个文件单文件自包含、可读。

runq.c 独有:`QuantizedTensor` 结构、`quantize()`/`dequantize()`、int8 matmul(Q8_0 分组对称量化:每 GS 个权重一组,组内按绝对值最大值定 scale,存 int8 + scale)。

> 注意:GPU 版 `run.cu` 基于 fp32 的 run.c,**没有量化的 GPU 版**;runq.c 是纯 CPU int8。

---

## 4. 核心数据结构

`Transformer` 结构体把三者合一(`run.cu:122`):

```c
typedef struct {
    Config config;              // 超参(蓝图)
    TransformerWeights weights; // 权重(只读模型)
    RunState state;             // 激活值缓冲(可写,流动的数据)
    int fd; float* data; ssize_t file_size;  // mmap 清理用
} Transformer;
```

### Config — 超参(.bin 开头 7 个 int)
```c
int dim;         // 主干隐藏维度          288
int hidden_dim;  // FFN 中间层维度        768
int n_layers;    // 解码层数              6
int n_heads;     // Query 头数            6
int n_kv_heads;  // KV 头数(<n_heads=GQA) 6
int vocab_size;  // 词表大小(负=不共享lm_head) 32000
int seq_len;     // 最大序列长度          256
```

### TransformerWeights — 权重(指针指向显存)
- `token_embedding_table` (vocab,dim) — 词嵌入表
- `rms_att_weight` / `rms_ffn_weight` (layer,dim) — 两处 RMSNorm 增益
- `wq/wk/wv/wo` — 注意力四矩阵(**每层独立**)
- `w1/w2/w3` — FFN 三矩阵(**每层独立**)
- `rms_final_weight` (dim,) — 最终 norm
- `wcls` — 分类头(本模型与 embedding 共享)

### RunState — 激活值缓冲(推理的"草稿纸")

**作用**:存放数据在网络里流动时每一步的中间结果。权重是不变的"模型",RunState 是流动的"数据"。

| 类别 | 字段 | 生命周期 | 作用 |
|------|------|---------|------|
| 主干状态 | `x` (dim) | 整个 forward | 贯穿 6 层的隐藏状态 |
| 临时草稿 | `xb,xb2,q,k,v,hb,hb2,att` | 用完即覆盖 | 算子间传中间结果 |
| 跨步记忆 | `key_cache,value_cache` | 整个生成过程 | KV Cache,累积历史 K/V |
| 输出 | `logits/logits_gpu` | 每步末尾 | 词表分数,拷回 CPU 采样 |

**设计要点**:所有 buffer 在 `malloc_run_state()` **启动时一次性 cudaMalloc**,之后每 token / 每层复用同一块显存,不在热路径反复申请。结构体在主机,但每个 `float*` 指向 GPU(`logits` 例外,在 CPU,因为采样在 CPU)。

---

## 4.5 三个 build 函数:初始化流程

`main()` 在跑推理前调用三个 `build_*`,分别把 Config+Weights+RunState、词表、采样器准备好。它们都只做"装载/分配",不做计算。

```
main()
├─ build_transformer(&t, model.bin)   →  t.config / t.weights / t.state
├─ build_tokenizer(&tok, tokenizer.bin, vocab_size)  →  tok.vocab / vocab_scores
└─ build_sampler(&s, vocab_size, temp, topp, seed)   →  s.temperature / topp / probindex
```

### build_transformer — 加载模型(权重 + 缓冲)

它本身极薄,只串起两步:

```c
void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, ...);  // ① 读 config + 加载权重
    malloc_run_state(&t->state, &t->config);                        // ② 按 config 开 RunState 缓冲
}
```

**① read_checkpoint**(详见第 5 节):
```
fopen → fread 28字节Config头 → 处理 shared_weights(vocab<0) → 求文件大小
     → mmap 整个文件 → (GPU) cudaMalloc 权重区并 cudaMemcpy 上显存
     → memory_map_weights() 把大指针按布局切成 token_embedding/wq/wk/.../wcls
```

**② malloc_run_state**:Config 读出来后才知道各 buffer 多大,这步**一次性 cudaMalloc** 所有激活缓冲(`x/xb/q/k/v/hb/att/...` 按 dim/hidden_dim,KV Cache 和 att 按 seq_len),之后整个推理复用、不再申请。`logits` 用 calloc 在 CPU(采样在 CPU)。

> 所以 `build_transformer` = **read_checkpoint(加载只读权重) + malloc_run_state(分配可写缓冲)**,正好对应"模型"和"草稿纸"两块。

### build_tokenizer — 加载词表

```c
void build_tokenizer(Tokenizer* t, char* path, int vocab_size) {
    t->vocab_size = vocab_size;              // ① 文件没存,从 config 传
    t->vocab/vocab_scores = malloc(...);     // ② 开数组
    t->sorted_vocab = NULL;                  //    字符串→id 索引懒加载(encode首次用才建)
    for(i<256) byte_pieces[...] = i;         // ③ 预填256个单字节兜底(UTF-8)
    fread(max_token_length); 循环读32000条:[score][len][bytes]  // ④ 读整张词表
}
```
细节见[第 8 节](#8-tokenizer-分词器)。只装载、不翻译。

### build_sampler — 初始化采样器

最简单,存几个采样参数 + 开一个 top-p 用的缓冲:

```c
void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, ull seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;   // -t,0=贪婪
    s->topp = topp;                 // -p,核采样阈值
    s->rng_state = seed;            // -s,随机种子
    s->probindex = malloc(vocab_size * sizeof(ProbIndex));  // top-p 排序用的临时缓冲
}
```
`Sampler` 结构体(run.cu:994):`vocab_size / probindex / temperature / topp / rng_state`。运行时 `sample()` 用这些参数从 logits 选 token。

### 三者对比

| | build_transformer | build_tokenizer | build_sampler |
|--|-------------------|-----------------|---------------|
| 加载来源 | 模型 `*.bin` | `tokenizer.bin` | 无(纯参数) |
| 产出 | config + 权重(GPU) + RunState 缓冲 | 词表(CPU) | 采样参数 + probindex |
| 加载方式 | mmap + cudaMemcpy | fread 逐项 | malloc |
| 数据落点 | 权重在 GPU,RunState 在 GPU | CPU | CPU |
| 是否做计算 | 否(只加载/分配) | 否 | 否 |

---

## 5. 模型文件 .bin 布局

`read_checkpoint` 先 `fread` 出 28 字节 Config 头,再 `mmap` 权重区拷到 GPU,最后 `memory_map_weights` 把一根大指针按布局切成各权重矩阵。

```
 stories15M.bin · 60,816,028 字节 (58.0 MiB) · n_layers=6
 ┌──────────────────────────────────────────────────────────────┐
 │ [0..28)  CONFIG 头 (7×int32)                                   │
 ╞══════════════════════════════════════════════════════════════╡
 │ 权重区(字节28起,连续 fp32,无名字无分隔)                     │
 │                                                                │
 │ token_embedding  [整块·全模型1份]  9,216,000 (35.2MiB)         │
 │                                                                │
 │ ╭── 以下每块内部按「6层首尾相接」存,L0→L5 各自独立 ──────╮    │
 │ │ wq  ┌L0┬L1┬L2┬L3┬L4┬L5┐ 单层288×288=82,944 / 共497,664 │    │
 │ │     └──┴──┴──┴──┴──┴──┘  ▲ w->wq + l*dim*dim 定位第l层  │    │
 │ │ wk  ┌L0┬L1┬L2┬L3┬L4┬L5┐ 同上                            │    │
 │ │ wv  ┌L0┬L1┬L2┬L3┬L4┬L5┐                                 │    │
 │ │ wo  ┌L0┬L1┬L2┬L3┬L4┬L5┐                                 │    │
 │ │ w1  ┌L0┬L1┬L2┬L3┬L4┬L5┐ 单层768×288=221,184/共1,327,104 │    │
 │ │ w2/w3 同 w1                                              │    │
 │ │ rms_att / rms_ffn ┌L0..L5┐ 单层288                       │    │
 │ ╰──────────────────────────────────────────────────────────╯ │
 │                                                                │
 │ rms_final_weight [整块] 288                                    │
 │ (RoPE freq_cis)  [老格式遗留,跳过] 12,288                      │
 │ wcls ──► 复用 token_embedding(偏移0,文件不单独存)            │
 └──────────────────────────────────────────────────────────────┘
   校验: 28 + 15,204,000 × 4 = 60,816,028 ✓
```

### 关键结构特征

**① type-major 布局(按权重类型分组,不是按层)**

文件里是 `[6个Wq][6个Wk][6个Wv][6个Wo][6个w1]...`,**6 个 W_Q 连续摆成一个大块**,放完所有 W_Q 才到 W_K。取第 l 层用块内偏移:
```c
matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);  // l=0→偏移0, l=1→82944, ...
```
原因:`export.py` 导出时 `for layer: write(layer.wq)` 把同类权重收集着写,加载端 `ptr += n_layers*dim*dim` 一句跳过整块,代码极简。

**② 每层 W_Q/W_K/W_V 互不共享** —— 6 层 = 6 套完全独立的权重。元素数 497,664 = 6×82,944 本身就是证据(共享的话只会存 1 份)。这是 Transformer 标准设计:浅层抓局部/语法,深层抓语义,需各自的变换。

**③ 谁分层 / 谁整块**

| 整块(全模型1份) | 分层(块内6个独立子块) |
|------------------|----------------------|
| token_embedding, rms_final, wcls(=embedding) | wq/wk/wv/wo, w1/w2/w3, rms_att/rms_ffn |

### 占比

```
 token_embedding ████████████████ 35.2 MiB (60%) ← 词表大,占一多半
 w1+w2+w3 (FFN)  ██████           15.2 MiB (26%)
 wq+wk+wv+wo     ████              7.6 MiB (13%)
```

---

## 6. 关键维度概念:dim / hidden_dim / head_size / seq_len

### dim(288)= token 嵌入向量长度 = 主干隐藏状态宽度

`dim` 既是**每个 token embedding 向量的长度**,也是**整个主干隐藏状态 `x` 的宽度**。嵌入是它的起点,之后一路保持这个尺寸流过所有层(注意力、残差、层间传递全是 288)。

```c
float* content_row = w->token_embedding_table + token * dim;  // 取一行=288个float
cudaMemcpy(x, content_row, dim*sizeof(float), ...);            // 直接当初始 x 用
```
> 嵌入维度必须 = 模型维度,因为嵌入出来直接当隐藏状态用。

### hidden_dim(768)= FFN 内部临时加宽的宽度

**不是**任何 token 向量的长度。只在 FFN 内部短暂出现:进 FFN 时 W1/W3 把 288 **升到** 768,做 SwiGLU 非线性,再用 W2 **降回** 288。出了 FFN 这个尺寸就消失。

```
 x[288] ─W1/W3─▶ [768] ─SwiGLU─▶ [768] ─W2─▶ x[288]
         升维         在宽处算激活        降回
```
意义:给模型更大的非线性表达空间。惯例 `hidden_dim ≈ 2.7×~4× dim`(本模型 768/288≈2.67×,SwiGLU 双路所以用 2.67× 而非 4×)。

| | dim (288) | hidden_dim (768) |
|--|-----------|------------------|
| 是 token 向量长度吗 | ✅ 是 | ❌ 不是 |
| 作用范围 | 全程 | 仅 FFN 内部 |
| 持续性 | 恒定 | 升上去马上降回 |

### head_size(48)= dim / n_heads

注意力把 288 维切成 6 个头,每个头 48 维(`288/6`)。

### seq_len(256)= 最大上下文长度

**就是上下文长度,且是最大值**。prompt + 生成的总 token 数不能超过它。两层原因:
1. **训练时**:位置编码(RoPE)是在这个长度范围内学的,超了效果差(Llama-3.1 的 rope scaling 就是为扩展它)。
2. **推理时**:KV Cache 和注意力分数缓冲按它预分配显存:
```c
cudaMalloc(&s->key_cache,   n_layers * seq_len * kv_dim * sizeof(float));
cudaMalloc(&s->value_cache, n_layers * seq_len * kv_dim * sizeof(float));
cudaMalloc(&s->att,         n_heads  * seq_len          * sizeof(float));
```
`main()` 里 `steps` 超过 seq_len 会被强制截断。seq_len 越大越吃显存——这是长上下文模型吃显存的根源。

---

## 7. hidden_states / 隐藏状态

`hidden_states` = 一个 token 在网络**内部**当前的向量表示。代码里就是那个贯穿全程的 `x`(长度 dim=288)。

- "hidden" = 网络内部中间表示,既非输入(token id)也非输出(logits)。
- "state" = 表示"这个 token 此刻被理解成了什么",每过一层被更新一次。

```c
float *x = s->x;                      // x 就是 hidden_states
cudaMemcpy(x, content_row, ...);      // 初始 = 词嵌入
for (l=0; l<n_layers; l++) {
    rmsnorm(s->xb, x, ...);           // 拿当前 hidden_states 算
    accum(x, s->xb2, dim);            // 注意力结果加回 → 更新
    accum(x, s->xb,  dim);            // FFN 结果加回 → 再更新
}
matmul(logits, x, w->wcls, ...);      // 最终 hidden_states → logits
```

| 名字 | 是什么 | 长度 |
|------|--------|------|
| token id | 整数(词表编号) | 标量 |
| embedding | token 的**初始**向量 | dim=288 |
| **hidden_states (x)** | token 在**中间层**的当前向量 | dim=288 |
| logits | **最终**词表分数 | vocab=32000 |

> embedding 是 hidden_states 的起点;hidden_states 是 embedding 过若干层加工后的样子;长度始终 288。`~/infer` 项目用 HuggingFace 风格直接叫 `hidden_states`,llama2.cu 精简叫 `x`,同一个东西。

---

## 8. Tokenizer 分词器

负责**字符串 ↔ token id 互转**,基于 **BPE(Byte Pair Encoding)**。

### 结构体(run.cu:773)
```c
typedef struct { char *str; int id; } TokenIndex;  // 一个"词→id"条目

typedef struct {
    char** vocab;                  // id→字符串。vocab[id]=第id个token的文字
    float* vocab_scores;           // 每个token的合并分数,BPE 编码时决定先合并谁
    TokenIndex *sorted_vocab;      // 按字符串排序,用于"字符串→id"二分查找
    int vocab_size;                // 32000
    unsigned int max_token_length; // 最长token字符数
    unsigned char byte_pieces[512];// 256个单字节兜底片段(UTF-8兜底)
} Tokenizer;
```

| 字段 | 作用 | 方向 |
|------|------|------|
| `vocab` | id→文字 | decode |
| `vocab_scores` | 合并优先级(总是先合并分数最高的相邻对) | encode |
| `sorted_vocab` | 字符串→id 二分查找 | encode |
| `byte_pieces` | 词表没收录的字符退回按字节输出 | decode |

两个出口:`encode`(字符串→token)、`decode`(token→字符串)。

### build_tokenizer 在干什么

**初始化分词器**:开内存 + 填 256 字节兜底 + 从 `tokenizer.bin` 读整张词表。只"装载",不做翻译。

```c
void build_tokenizer(Tokenizer* t, char* path, int vocab_size) {
    t->vocab_size = vocab_size;              // ① 文件没存,从 config 传(作者吐槽 sigh)
    t->vocab        = malloc(...);           // ② 开指针数组(还没填内容)
    t->vocab_scores = malloc(...);
    t->sorted_vocab = NULL;                  //    "字符串→id"索引懒加载(encode首次用才建)
    for (i<256) { byte_pieces[i*2]=i; ... }  // ③ 预填256个单字节字符串(UTF-8兜底)
    fread(&t->max_token_length, ...);        // ④ 读文件:先读最长token长度
    for (i<vocab_size) {
        fread(vocab_scores+i, float);        //    读分数
        fread(&len, int); vocab[i]=malloc(len+1); fread(vocab[i], len);  // 读长度+字符串
        vocab[i][len]='\0';
    }
}
```

**tokenizer.bin 格式**(被这段代码揭示):
```
[max_token_length:int] 然后 32000 条 [score:float][len:int][bytes:变长]
```

设计细节:`sorted_vocab` 懒加载(decode 用不上,省启动开销);`byte_pieces` 提前填好做 UTF-8 兜底;每个 vocab[i] 按 len 单独 malloc(token 变长)。

### token 的 score 是什么 & BPE 合并过程

`vocab_scores[id]` 是每个 token 的**合并优先级**,本质是 SentencePiece 训练时学到的**(对数)概率分数**:负数,越接近 0 = 越高频/越优先。

`encode` 的 BPE 是**贪心合并**:先把字符串拆成单字符,然后每一轮**扫描所有相邻对**,查出每对合并后那个 token 的预存 score,**选 score 最高的合并一次**,直到无对可合。

```c
while (1) {
    float best_score = -1e10; int best_idx = -1;
    for (每对相邻 token i,i+1) {
        id = 查词表(vocab[i] + vocab[i+1]);        // 这对能合成已知 token 吗
        if (id != -1 && vocab_scores[id] > best_score) {  // 比 score,谁高选谁
            best_score = vocab_scores[id]; best_idx = i;
        }
    }
    if (best_idx == -1) break;                     // 没有可合并的 → 结束
    合并 best_idx 这一对;                           // 一轮只合一对
}
```

> 关键:`score` 是"合并结果 token 在词表里的固定分数",**查表得到、非临时计算**。贪心比的就是这些固定值。

**实跑追踪**(本仓库给 `encode` 加了 `TRACE_BPE` 开关):
```bash
bash scripts/run.sh -T -i "Once upon a time"   # -T 开启 BPE 过程打印
```
真实输出(stories15M 词表):
```
初始: [_][O][n][c][e][_][u][p][o][n][_][a][_][t][i][m][e]   (18 tokens)
第1步: [_]+[t]  -> [_t]     score=-1       ← 分数最高,先合
第3步: [o]+[n]  -> [on]     score=-6
...
第12步:[_up]+[on] -> [_upon] score=-2242
第13步:[_On]+[ce] -> [_Once] score=-8779   ← 分数最低,最后合
最终: [_Once][_upon][_a][_time]  →  id: [1, 9038, 2501, 263, 931]
```
现象:score 列单调递减(每轮选当前最高),**高频小片段(`_t`/`on`)先合,罕见完整词(`_Once`)最后才拼出**。BPE 合并**只对 prompt 做一次**;生成阶段输出的本就是 token id,无需再 encode。

---

## 9. 运行时主流程:generate / forward / sample

真正的"运行"从 `generate()` 开始(三个 build 都只是装载)。本节讲生成主循环,以及它每步调用的 `forward` 和 `sample`。

### 9.1 generate() 主干流程

```
generate(transformer, tokenizer, sampler, prompt, steps)
├─ 1. 分词    encode(prompt) → prompt_tokens[], num_prompt_tokens
├─ 2. 初始化  token = prompt_tokens[0];  pos = 0
├─ 3. 主循环 while (pos < steps):
│      a. forward(token, pos) → logits        ← 算一次完整前向
│      b. 下一个 token:
│           还在 prompt 内 → 用 prompt_tokens[pos+1]   (不采样)
│           已过 prompt    → sample(logits)            (采样)
│      c. pos++
│      d. next==1 (BOS) → break               ← 序列结束
│      e. decode(token→文字) 并打印            ← 流式输出
│      f. token = next                         ← 喂回,自回归
└─ 4. 收尾   打印 tok/s,free
```

```c
void generate(...) {
    encode(tok, prompt, 1/*BOS*/, 0, prompt_tokens, &num_prompt_tokens);  // 分词
    int token = prompt_tokens[0], pos = 0, next;
    while (pos < steps) {
        float* logits = forward(t, token, pos);          // a
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];  // b: prompt阶段,不采样
        else                             next = sample(s, logits);       //    生成阶段,采样
        pos++;                                            // c
        if (next == 1) break;                             // d: BOS=结束符
        safe_printf(decode(tok, token, next));            // e: 流式打印
        token = next;                                     // f: 自回归喂回
    }
    // 4. tok/s 统计(计时从第2步起,排除较慢的首步)
}
```

三个要点:**① 自回归**——`token=next` 把这步的输出喂作下步输入,一个接一个滚。**② 流式输出**——每个 token 立即 decode+打印+fflush,所以文字一个个蹦出来。**③ 何时停**——`pos>=steps`(到 -n 上限)或 `next==1`(采到 BOS)。

### 9.2 forward 是什么(不是某个"阶段")

`forward` 是**神经网络的一次完整前向传播**:把 1 个 token 从头算到尾,吐出 32000 个 logits。它**包含**了所有计算步骤(嵌入→6层解码→最终norm→lm_head),细节见 `transformer-inference.md`。

- forward 是"动词"(算一次);**prefill / decode 只是调用它的两种场景**,用的是同一个 forward。
- `forward(t, token, pos)` 签名只接收**单个 token** → 这个 demo 逐 token 串行,无批处理能力。

### 9.3 prefill vs decode:共用一个循环

**有 prefill 的"逻辑",没有 prefill 的"优化"。** 二者共用 9.1 的循环,区别只在「下一个 token 从哪来」:

| | prompt 阶段(prefill) | 生成阶段(decode) |
|--|----------------------|------------------|
| pos 范围 | `< num_prompt_tokens-1` | 之后 |
| forward | ✅ 照算(为填 KV Cache) | ✅ |
| 下一个 token | 用已知 prompt token | `sample()` 采样 |
| logits | **算了但丢弃** | 用来采样 |

| | 真正的 prefill(如 ~/infer) | llama2.cu |
|--|---------------------------|-----------|
| 处理 prompt | 一次性**并行**整个 prompt | **逐 token 串行** |
| attention kernel | 专门 prefill flash-attn | 与 decode 共用 |
| lm_head | prompt 只在最后位置算 | **每步都算(prompt 阶段冗余)** |

### 9.4 lm_head:把隐藏状态变成词表分数

forward 的最后一步,代码里叫 `wcls`,是 `(vocab, dim)=(32000,288)` 的分类矩阵(Language Model Head):

```c
matmul(s->logits_gpu, x, w->wcls, dim, vocab_size);  // x[288] → logits[32000]
```

- **为什么算全部 32000 个**:采样要在所有候选词里选,必须给每个词都打分才能形成概率分布——这在"要采样的步"是必需的,不是浪费。
- **真正的冗余**:prompt 阶段每步也算了 lm_head 却丢掉(下个词已知)。优化引擎会在 prefill 只对最后位置算。
- **本模型 wcls 与 token_embedding 共享**(weight tying,见第 5 节):输入"词→向量"和输出"向量→词"互逆,复用一套权重省 35MB。
- lm_head 是单次 forward 里最大的矩阵乘(vocab 32000 ≫ dim 288)。

### 9.5 Sampler:从 logits 选 token 的决策抽象

`Sampler`(run.cu:994)封装"如何从 logits 选下一个词":策略参数(`temperature`/`topp`)+ 随机状态(`rng_state`)+ 工作缓冲(`probindex`)。**每个生成步采样一次,产出 1 个 token**:

```c
int sample(Sampler* s, float* logits) {
    if (s->temperature == 0) return sample_argmax(...);   // ① 贪婪:取最高分(确定性)
    for(...) logits[q] /= s->temperature;                 //    温度缩放
    softmax(logits, vocab);                               //    → 概率
    float coin = random_f32(&s->rng_state);               //    掷骰子
    return s->topp>0 && s->topp<1 ? sample_topp(...)      // ② top-p 核采样
                                  : sample_mult(...);      // ③ 全分布多项式采样
}
```

分工:**Transformer 算分数(logits)→ Sampler 拍板选词 → Tokenizer 译回文字**。

> 数量关系:生成阶段 **每步 = 1 次 forward + 1 次 sample → 1 个 token**;prompt 阶段 forward 照跑但跳过 sample(下个词已知)。

---

## 10. 观测日志(read_checkpoint + memory_map_weights)

本仓库在加载阶段加了观测日志(走 stderr,不污染 stdout 生成文本),启动即打印:

**Config 头**:dim/hidden_dim/n_layers/n_heads/n_kv_heads/head_size/vocab/seq_len/shared_weights/file_size。

**权重布局表**:每个权重块的形状、总元素、是否分层(`L×`=按层细分 / `—`=整块)、单层元素、偏移。例:
```
名称                形状              总元素    分层   单层元素    偏移
token_embedding...  (vocab, dim)     9216000   —     (整块)         0
wq                  (layer, dim, Qd)  497664   L×      82944    9217728
w1                  (layer, hidden..) 1327104  L×     221184   11210112
...
wcls                (vocab, dim)     9216000   —     (整块)         0  ← 与embedding共享
```

`L×` 列直接体现:wq/wk/wv/wo/w1/w2/w3/rms_att/rms_ffn 都是块内按 6 层细分、各层独立;token_embedding/rms_final/wcls 是整块共享。换更大的模型(如 stories110M)跑一遍,所有数字相应变化,结构不变。

代码位置:`Config` 注释 @60,`read_checkpoint` 日志 @~280,`memory_map_weights` 日志 @~246。

---

## 相关文档
- 推理流程逐阶段拆解:`docs/transformer-inference.md`(含 Mermaid + ASCII 图)
- 交互网页版:`docs/transformer-inference.html`
- 编译运行指南:`BUILD_AND_RUN.md`
