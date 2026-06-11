#!/usr/bin/env bash
# 下载 TinyStories 预训练权重
#   bash scripts/download_model.sh        # 默认 15M
#   bash scripts/download_model.sh 110M   # 110M (更大更连贯)
#   bash scripts/download_model.sh 42M
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

SIZE="${1:-15M}"
BASE_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main"

case "${SIZE}" in
  15M|42M|110M) FILE="stories${SIZE}.bin" ;;
  *) echo "未知尺寸: ${SIZE} (可选: 15M / 42M / 110M)"; exit 1 ;;
esac

if [ -f "${FILE}" ]; then
  echo ">> ${FILE} 已存在,跳过下载。"
  exit 0
fi

echo ">> 下载 ${FILE} ..."
if command -v wget >/dev/null; then
  wget -q --show-progress "${BASE_URL}/${FILE}" -O "${FILE}"
elif command -v curl >/dev/null; then
  curl -L --progress-bar "${BASE_URL}/${FILE}" -o "${FILE}"
else
  echo "错误: 需要 wget 或 curl"; exit 1
fi
echo ">> 完成: ${ROOT_DIR}/${FILE}"
