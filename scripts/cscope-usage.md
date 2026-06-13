# gen_cscope.sh 使用说明

用 cscope 给本项目建立代码符号索引,方便在 `run.cu` / `run.c` / `runq.c` 等文件间**跳转定义、查找调用、追踪符号**。

> **为什么需要这个脚本**:cscope 默认的扩展名白名单只含 `.c/.h/.cpp` 等,**不识别 `.cu/.cuh`(CUDA)**,递归扫描时会直接跳过 `run.cu`。本脚本显式生成文件清单(`cscope.files`)把 `.cu/.cuh` 纳入,从而能索引到 `forward`、各 CUDA kernel 等符号。

---

## 1. 前置条件

需要安装 cscope:
```bash
sudo apt install cscope        # Debian/Ubuntu
# 或 yum install cscope        # CentOS/RHEL
```
脚本会自动检查,未安装会报错提示。

---

## 2. 生成索引

在项目任意位置运行(脚本会自动定位到项目根目录):
```bash
bash scripts/gen_cscope.sh
```

它会:
1. 清掉旧索引(`cscope.out` / `cscope.in.out` / `cscope.po.out` / `cscope.files`)
2. 用 `find` 生成文件清单 `cscope.files`,**显式包含** `.cu/.cuh/.c/.h/.cpp/.hpp/.cc`,排除 `build/` 和 `.git/`
3. `cscope -bqk` 建反向索引(`-b` 只建库、`-q` 反向索引加速、`-k` 不含系统头)
4. 自检:打印 `forward` 的定义,确认 `run.cu` 已被索引

输出示例:
```
>> 索引 6 个文件:
   ./run.c
   ./run.cu
   ...
>> 完成: .../cscope.out
>> 自检 (查 forward 的定义):
   run.cu forward 690 float * forward(Transformer* transformer, int token, int pos) {
```

> 改动代码后重新运行本脚本即可刷新索引。生成的 `cscope.*` 已在 `.gitignore` 中,不会被提交。

---

## 3. 命令行查询(不进交互界面)

`-d` = 用已有索引不重建,`-L` = 单次查询模式,`-N` 指定查询类型:

| 命令 | 作用 |
|------|------|
| `cscope -dL -0 <符号>` | 查符号出现的所有位置 |
| `cscope -dL -1 <符号>` | 查**定义** |
| `cscope -dL -2 <符号>` | 查该函数**调用了谁** |
| `cscope -dL -3 <符号>` | 查**谁调用了**该函数 |
| `cscope -dL -6 <文本>`  | 全局正则/文本搜索 |
| `cscope -dL -7 <文件>`  | 查找文件 |
| `cscope -dL -8 <文件>`  | 查谁 #include 了该文件 |

示例:
```bash
cscope -dL -1 forward                       # forward 定义在哪
cscope -dL -3 multi_head_attention          # 谁调用了它
cscope -dL -1 multi_head_attention_kernel   # CUDA kernel 定义(只在 run.cu)
```

---

## 4. 交互界面

```bash
cscope -d        # 进入交互 TUI(用已有索引)
```
- 上下方向键在输入栏的各查询类型间移动(Find this definition / Find functions called by / ...)
- 输入符号回车,结果列表里按对应编号跳转
- `Ctrl+D` 退出

---

## 5. vim / neovim 集成

```vim
" 在项目根目录打开 vim 后:
:cs add cscope.out

" 常用快捷查询:
:cs find g forward          " 跳到定义 (g = global definition)
:cs find c forward          " 谁调用了 forward (c = callers)
:cs find s forward          " 该符号所有出现 (s = symbol)
```
建议在 `.vimrc` 里加自动加载:
```vim
if filereadable("cscope.out")
    cs add cscope.out
endif
```

---

## 6. 常见问题

| 现象 | 原因 / 解决 |
|------|------------|
| `run.cu` 的符号查不到 | 没用本脚本,而是 `cscope -R`(默认跳过 .cu)→ 改用 `bash scripts/gen_cscope.sh` |
| `cscope: command not found` | 未安装 → `sudo apt install cscope` |
| 改了代码后跳转位置不对 | 索引过期 → 重新运行脚本 |
| 想索引更多文件类型 | 编辑脚本里 `find` 的 `-name '*.xxx'` 列表 |

---

## 7. 相关

- 脚本本体:`scripts/gen_cscope.sh`
- 其它脚本:`build.sh`(编译)、`download_model.sh`(下模型)、`run.sh`(运行推理,`-T` 开 BPE 追踪)
- 代码导读:`docs/code-walkthrough.md`
