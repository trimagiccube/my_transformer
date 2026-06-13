#!/usr/bin/env bash
# 重建 cscope 索引(显式包含 .cu/.cuh —— cscope 默认不识别 CUDA 后缀)
#   bash scripts/gen_cscope.sh
#
# 之后用法:
#   cscope -d                          # 进交互界面
#   cscope -dL -1 <symbol>             # 查某符号的定义
#   cscope -dL -3 <symbol>             # 查某函数被谁调用
# vim 用户::cs add cscope.out 即可跳转
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

command -v cscope >/dev/null || { echo "错误: 未安装 cscope (sudo apt install cscope)"; exit 1; }

# 清掉旧库
rm -f cscope.out cscope.in.out cscope.po.out cscope.files

# 生成文件清单:显式列出 .cu/.cuh 及常规 C/C++ 源文件,排除 build 产物
find . \( -name '*.cu'  -o -name '*.cuh' \
       -o -name '*.c'   -o -name '*.h'   \
       -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cc' \) \
     -not -path './build/*' -not -path './.git/*' \
     | sort > cscope.files

n=$(wc -l < cscope.files)
echo ">> 索引 ${n} 个文件:"
sed 's/^/   /' cscope.files

# -b 只建库不进交互, -q 建反向索引(加速), -k 不索引系统头
cscope -bqk
echo ">> 完成: ${ROOT_DIR}/cscope.out"
echo ">> 自检 (查 forward 的定义):"
cscope -dL -1 forward 2>/dev/null | sed 's/^/   /' || true
