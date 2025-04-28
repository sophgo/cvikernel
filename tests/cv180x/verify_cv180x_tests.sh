#!/bin/bash

# 该脚本用于验证所有修改后的CV180X测试文件

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 需要验证的文件列表（使用相对路径）
FILES=(
  "tiu_add_cv180x.c"
  "tiu_and_cv180x.c"
  "tiu_avgpool_cv180x.c"
  "tiu_conv_cv180x.c"
  "tiu_matmul_cv180x.c"
  "tiu_max_cv180x.c"
  "tiu_maxpool_cv180x.c"
  "tiu_min_cv180x.c"
  "tiu_mul_cv180x.c"
  "tiu_normalize_cv180x.c"
  "tiu_or_cv180x.c"
  "tiu_quantize_cv180x.c"
  "tiu_relu_cv180x.c"
  "tiu_sub_cv180x.c"
  "tiu_xor_cv180x.c"
)

# 创建build目录
mkdir -p "${ROOT_DIR}/build/cv180x"

# 测试结果统计
TOTAL_FILES=0
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# 验证单个文件
verify_file() {
  local file=$1
  
  echo "==============================================="
  echo "验证文件: $file"
  
  TOTAL_FILES=$((TOTAL_FILES + 1))
  
  local source_file="${SCRIPT_DIR}/${file}"
  local output_file="${ROOT_DIR}/build/cv180x/${file%.c}"
  
  if [ ! -f "$source_file" ]; then
    echo -e "${RED}错误: 文件不存在${NC}"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi
  
  # 编译文件
  echo "编译中..."
  gcc -I"${ROOT_DIR}/cvikernel/include" "$source_file" -o "$output_file" -lm
  
  if [ $? -ne 0 ]; then
    echo -e "${RED}编译失败${NC}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return
  fi
  
  # 运行测试
  echo "运行测试..."
  "$output_file" > /dev/null 2>&1
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}测试通过${NC}"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo -e "${RED}测试失败${NC}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

echo "==============================================="
echo "开始验证CV180X测试文件"
echo "==============================================="

# 遍历验证每个文件
for file in "${FILES[@]}"; do
  verify_file "$file"
done

# 打印总结报告
echo "==============================================="
echo "验证完成"
echo "==============================================="
echo "总文件数: $TOTAL_FILES"
echo -e "${GREEN}通过: $PASS_COUNT${NC}"
echo -e "${RED}失败: $FAIL_COUNT${NC}"
echo -e "跳过: $SKIP_COUNT"
echo "==============================================="

# 如果所有文件都通过，则返回成功状态
if [ $PASS_COUNT -eq $TOTAL_FILES ]; then
  echo -e "${GREEN}所有文件验证通过!${NC}"
  exit 0
else
  echo -e "${RED}部分文件验证失败!${NC}"
  exit 1
fi 