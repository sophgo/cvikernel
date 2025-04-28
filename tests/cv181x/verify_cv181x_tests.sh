#!/bin/bash

# 该脚本用于编译和运行所有CV181X测试文件

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 设置文件列表 - 根据目录中实际存在的文件更新
FILES=(
  "tiu_add_cv181x.c"
  "tiu_conv_cv181x.c"
  "tiu_ge_cv181x.c"
  "tiu_lookup_table_cv181x.c"
  "tiu_mac_cv181x.c"
  "tiu_max_cv181x.c"
  "tiu_min_cv181x.c"
  "tiu_min_pooling_cv181x.c"
  "tiu_mul_cv181x.c"
  "tiu_mul_qm_cv181x.c"
  "tiu_or_cv181x.c"
  "tiu_shift_cv181x.c"
  "tiu_sub_cv181x.c"
  "tiu_xor_cv181x.c"
)

# 统计变量
total_files=${#FILES[@]}
passed=0
failed=0
skipped=0

# 创建build目录
mkdir -p "${ROOT_DIR}/build/cv181x"

# 打印标题
echo "==============================================="
echo "开始验证CV181X测试文件"
echo "==============================================="

# 遍历所有文件
for file in "${FILES[@]}"; do
  echo "==============================================="
  echo "验证文件: $file"
  
  source_file="${SCRIPT_DIR}/${file}"
  output_file="${ROOT_DIR}/build/cv181x/${file%.c}"
  
  # 检查源文件是否存在
  if [ ! -f "$source_file" ]; then
    echo -e "${RED}源文件不存在，跳过${NC}"
    ((skipped++))
    continue
  fi
  
  # 编译文件
  echo "编译中..."
  gcc -I"${ROOT_DIR}/cvikernel/include" "$source_file" -o "$output_file" -lm -Werror
  
  # 检查编译结果
  if [ $? -ne 0 ]; then
    echo -e "${RED}编译失败${NC}"
    ((failed++))
    continue
  fi
  
  # 运行测试
  echo "运行测试..."
  "$output_file"
  
  # 检查运行结果
  if [ $? -ne 0 ]; then
    echo -e "${RED}测试失败${NC}"
    ((failed++))
  else
    echo -e "${GREEN}测试通过${NC}"
    ((passed++))
  fi
done

# 打印统计信息
echo "==============================================="
echo "验证完成"
echo "==============================================="
echo "总文件数: $total_files"
echo -e "${GREEN}通过: $passed${NC}"
echo -e "${RED}失败: $failed${NC}"
echo "跳过: $skipped"
echo "==============================================="

# 检查是否所有文件都通过
if [ $failed -eq 0 ]; then
  echo -e "${GREEN}所有文件验证通过!${NC}"
  exit 0
else
  echo -e "${RED}有文件验证失败!${NC}"
  exit 1
fi 