# llama2.cu —— 编译、依赖与推理运行指南

> 本文档针对 GPU(CUDA)推理。项目核心是单文件 `run.cu`,是 karpathy `llama2.c` 的 CUDA 移植版。

---

## 1. 运行时 / 编译依赖

### 必需(GPU 推理)
| 依赖 | 说明 | 本机已验证版本 |
|------|------|----------------|
| NVIDIA GPU | 计算能力 sm_52 及以上即可 | RTX 4060 Ti (sm_89) / RTX 5060 (sm_120) |
| NVIDIA 驱动 | 支持你的 CUDA 版本 | 590.48.01 |
| CUDA Toolkit | 提供 `nvcc` 和 cuBLAS;建议 12.x | 12.6 |
| cuBLAS | 随 CUDA Toolkit 安装,矩阵乘法用 | 随 12.6 |
| C 编译器 | gcc/clang,编 CPU 版用;CUDA 版由 nvcc 调用 | gcc |

> 说明:GEMM(矩阵乘)调用 **cuBLAS**,其余算子(RMSNorm / RoPE / softmax / attention / 采样)是项目自写的 CUDA kernel。

### 可选(仅训练 / 导出模型时需要,纯推理不需要)
- Python 3 + PyTorch + numpy + sentencepiece(见 `requirements.txt`)
- 仅当你要自己训练模型或用 `export.py` 转换 Meta 官方 Llama-2 权重时才用得到。

### 关于本机两块卡的注意点
- **RTX 4060 Ti (sm_89)**:CUDA 12.6 原生支持,直接编译运行。
- **RTX 5060 (sm_120 / Blackwell)**:CUDA 12.6 无法生成原生 SASS,靠 **PTX JIT** 运行(首次启动略慢,功能正常)。如需原生支持请升级到 CUDA 12.8+。

---

## 2. 编译

### 方式一:用 Makefile(推荐)
```bash
cd ~/llama2.cu
make runcuda          # 生成 GPU 可执行文件 ./runcuda
```
等价于:
```bash
nvcc -DUSE_CUDA -O3 -o runcuda run.cu -lm -lcublas
```

其它构建目标:
| 目标 | 产物 | 用途 |
|------|------|------|
| `make runcuda` | `runcuda` | **GPU 推理(本指南主角)** |
| `make rundebugcuda` | `runcuda` | GPU 调试版(`-g`) |
| `make run` | `run` / `runq` | CPU fp32 / int8 推理 |
| `make runfast` | `run` | CPU `-Ofast` 优化版 |

### 方式二:用本仓库脚本
```bash
bash scripts/build.sh         # 默认编 runcuda
bash scripts/build.sh cpu     # 编 CPU 版 run
```

---

## 3. 获取模型权重

项目自带 `tokenizer.bin`,但**不含模型权重**,需下载:

```bash
bash scripts/download_model.sh            # 默认下 stories15M(58MB)
bash scripts/download_model.sh 110M       # 下 stories110M(更大更好,~420MB)
bash scripts/download_model.sh 42M        # 下 stories42M
```

可选权重(TinyStories 训练的小模型):`stories15M.bin` / `stories42M.bin` / `stories110M.bin`

---

## 4. 运行推理

### 最简单
```bash
./runcuda stories15M.bin
```

### 用脚本(自动选卡、带常用参数)
```bash
bash scripts/run.sh                                   # 默认参数跑 stories15M
bash scripts/run.sh -m stories110M.bin -n 256 -t 0.8 -i "Once upon a time"
bash scripts/run.sh -g 1                              # 用 GPU 1 (RTX 5060)
```

### 命令行参数(`run <checkpoint> [options]`)
| 参数 | 含义 | 默认 |
|------|------|------|
| `-n <int>` | 生成 token 数,0 = max_seq_len | 256 |
| `-t <float>` | 温度 [0,inf),0 = 贪婪 | 1.0 |
| `-p <float>` | top-p (nucleus) 采样 [0,1] | 0.9 |
| `-s <int>` | 随机种子 | 当前时间 |
| `-i <string>` | 输入 prompt | 无 |
| `-z <string>` | 自定义 tokenizer 路径 | tokenizer.bin |
| `-m <string>` | 模式:`generate` 或 `chat` | generate |
| `-y <string>` | chat 模式的 system prompt | 无 |

### 指定 GPU
```bash
CUDA_VISIBLE_DEVICES=0 ./runcuda stories15M.bin   # 用 RTX 4060 Ti
CUDA_VISIBLE_DEVICES=1 ./runcuda stories15M.bin   # 用 RTX 5060
```

### 例子
```bash
# 生成一段故事
./runcuda stories15M.bin -n 256 -i "Once upon a time" -t 0.8

# 贪婪解码(确定性输出)
./runcuda stories15M.bin -t 0 -i "The little robot"

# 对话模式
./runcuda stories15M.bin -m chat -y "You are a helpful assistant."
```

程序结束会在 stderr 打印 `achieved tok/s: ...` 性能数据。

---

## 5. 常见问题

- **`nvcc: command not found`**:CUDA Toolkit 未安装或未加入 PATH。检查 `/usr/local/cuda/bin`。
- **`cannot find -lcublas`**:cuBLAS 未装,随 CUDA Toolkit 安装即可。
- **RTX 5060 首次运行慢**:PTX JIT 编译,正常现象;升级 CUDA 12.8+ 可消除。
- **想跑真 Llama-2-7B**:用 `export.py` 转换,但 fp32 下 7B 需 ~28GB 显存,本机 8GB 卡放不下,建议用小模型学习。
