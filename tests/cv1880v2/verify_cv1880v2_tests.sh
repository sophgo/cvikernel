#!/bin/bash

# 该脚本用于编译和运行所有修改过的CV1880V2测试文件

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 测试文件列表（使用相对路径）
FILES=(
  "tiu_add_cv1880v2.c"
  "tiu_avgpool_cv1880v2.c"
  "tiu_batchnorm_cv1880v2.c"
  "tiu_conv_cv1880v2.c"
  "tiu_divide_cv1880v2.c"
  "tiu_eq_cv1880v2.c"
  "tiu_ge_cv1880v2.c"
  "tiu_lt_cv1880v2.c"
  "tiu_matmul_cv1880v2.c"
  "tiu_max_cv1880v2.c"
  "tiu_maxpool_cv1880v2.c"
  "tiu_min_cv1880v2.c"
  "tiu_multiply_cv1880v2.c"
  "tiu_quantize_cv1880v2.c"
  "tiu_relu_cv1880v2.c"
  "tiu_sigmoid_cv1880v2.c"
  "tiu_softmax_cv1880v2.c"
  "tiu_subtract_cv1880v2.c"
)

# 成功和失败计数器
success_count=0
fail_count=0

# 创建build目录
mkdir -p "${ROOT_DIR}/build/cv1880v2"

# 遍历编译和测试每个文件
for file in "${FILES[@]}"; do
  echo "==============================================="
  echo "测试文件: $file"
  
  source_file="${SCRIPT_DIR}/${file}"
  output_file="${ROOT_DIR}/build/cv1880v2/${file%.c}"
  
  if [ -f "$source_file" ]; then
    echo "编译 $file..."
    gcc -I"${ROOT_DIR}/cvikernel/include" "$source_file" -o "$output_file" -lm
    
    if [ $? -eq 0 ]; then
      echo "运行 $file..."
      "$output_file" > /dev/null 2>&1
      
      if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 测试成功${NC}"
        ((success_count++))
      else
        echo -e "${RED}❌ 测试失败${NC}"
        ((fail_count++))
      fi
    else
      echo -e "${RED}❌ 编译失败${NC}"
      ((fail_count++))
    fi
    echo ""
  else
    echo -e "${RED}文件不存在: $source_file${NC}"
    ((fail_count++))
  fi
done

echo "==============================================="
echo "测试结果统计:"
echo -e "${GREEN}✅ 成功: $success_count${NC}"
echo -e "${RED}❌ 失败: $fail_count${NC}"
echo "总计: $((success_count + fail_count))"
echo "==============================================="

if [ $fail_count -eq 0 ]; then
  echo -e "${GREEN}所有测试文件已成功修改并通过测试!${NC}"
  exit 0
else
  echo -e "${RED}有 $fail_count 个测试文件修改失败，请检查。${NC}"
  exit 1
fi 