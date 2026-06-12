#!/usr/bin/env bash
# 运行 GPU 推理
#   bash scripts/run.sh                                  # 默认 stories15M, GPU 0
#   bash scripts/run.sh -g 1                             # 用 GPU 1
#   bash scripts/run.sh -T                               # 开启 BPE 分词过程追踪
#   bash scripts/run.sh -m stories110M.bin -n 256 -t 0.8 -i "Once upon a time"
#
# 选项:
#   -g <int>     使用的 GPU 序号 (默认 0)
#   -m <file>    模型权重文件 (默认 stories15M.bin)
#   -T           开启 BPE 分词追踪 (设置 TRACE_BPE=1,打印 encode 合并过程)
#   其余参数 (-n -t -p -s -i -z) 原样透传给 runcuda
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

GPU=0
MODEL="stories15M.bin"
TRACE=0
PASS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g) GPU="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    -T) TRACE=1; shift ;;
    *)  PASS+=("$1"); shift ;;
  esac
done

# 没传任何推理参数时,给一组合理默认值
if [ ${#PASS[@]} -eq 0 ]; then
  PASS=(-n 1000 -t 0.8 -i "Once upon a time")
fi

[ -x ./runcuda ] || { echo "错误: 未找到 ./runcuda,请先运行 bash scripts/build.sh"; exit 1; }
[ -f "${MODEL}" ] || { echo "错误: 未找到模型 ${MODEL},请先运行 bash scripts/download_model.sh"; exit 1; }

# 组装环境变量:仅当 -T 开启时设置 TRACE_BPE=1 传给 runcuda
ENVV=()
[ "${TRACE}" = "1" ] && ENVV+=("TRACE_BPE=1")

echo ">> GPU=${GPU}  模型=${MODEL}  TRACE_BPE=${TRACE}  参数=${PASS[*]}"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES="${GPU}" env "${ENVV[@]}" ./runcuda "${MODEL}" "${PASS[@]}"
