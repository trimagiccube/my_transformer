#!/usr/bin/env bash
# 编译 llama2.cu
#   bash scripts/build.sh        # 编 GPU 版 (runcuda)
#   bash scripts/build.sh cpu    # 编 CPU 版 (run / runq)
#   bash scripts/build.sh debug  # 编 GPU 调试版
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

TARGET="${1:-gpu}"

case "${TARGET}" in
  gpu)
    command -v nvcc >/dev/null || { echo "错误: 找不到 nvcc,请安装 CUDA Toolkit"; exit 1; }
    echo ">> 编译 GPU 版 (runcuda) ..."
    make runcuda
    echo ">> 完成: ${ROOT_DIR}/runcuda"
    ;;
  debug)
    command -v nvcc >/dev/null || { echo "错误: 找不到 nvcc,请安装 CUDA Toolkit"; exit 1; }
    echo ">> 编译 GPU 调试版 (runcuda, -g) ..."
    make rundebugcuda
    echo ">> 完成: ${ROOT_DIR}/runcuda (debug)"
    ;;
  cpu)
    echo ">> 编译 CPU 版 (run / runq) ..."
    make runfast
    echo ">> 完成: ${ROOT_DIR}/run, ${ROOT_DIR}/runq"
    ;;
  *)
    echo "用法: bash scripts/build.sh [gpu|cpu|debug]"
    exit 1
    ;;
esac
