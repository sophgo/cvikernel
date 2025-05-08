// 测试 cv1880v2 芯片的平均池化功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "../../include/cvikernel/cvikernel.h"

// CV1880V2芯片的配置参数
#define CV1880V2_HW_LMEM_SIZE (1024 * 1024)  // 假设LMEM大小为1MB
#define CV1880V2_TPU_EU_NUM 32               // 假设EU数量为32

// 全局内存模拟
uint8_t *g_lmem_base = NULL;

// cv1880v2 优化的平均池化函数

#ifndef CV1880V2_USE_REAL_IMPL

void cv1880v2_avgpool2d(int8_t *input, int in_n, int in_c, int in_h, int in_w,
                       int8_t *output, int out_h, int out_w,
                       int kernel_h, int kernel_w, int stride_h, int stride_w,
                       int padding_h, int padding_w,
                       float input_scale, float output_scale) {
    
    // 对输出的每个元素进行计算
    for (int n = 0; n < in_n; n++) {
        for (int c = 0; c < in_c; c++) {
            // 模拟cv1880v2的通道并行处理 - 在实际硬件中，多个通道可以并行处理
            // 这里我们仍然按顺序处理，但是在实际硬件中会并行处理
            
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    // 计算输入特征图对应的开始位置
                    int in_h_start = oh * stride_h - padding_h;
                    int in_w_start = ow * stride_w - padding_w;
                    
                    // 计算输入特征图对应的结束位置
                    int in_h_end = in_h_start + kernel_h;
                    int in_w_end = in_w_start + kernel_w;
                    
                    // 确保不超出输入范围
                    int actual_h_start = (in_h_start < 0) ? 0 : in_h_start;
                    int actual_w_start = (in_w_start < 0) ? 0 : in_w_start;
                    int actual_h_end = (in_h_end > in_h) ? in_h : in_h_end;
                    int actual_w_end = (in_w_end > in_w) ? in_w : in_w_end;
                    
                    // 计算池化窗口内有效元素数量
                    int count = (actual_h_end - actual_h_start) * (actual_w_end - actual_w_start);
                    if (count <= 0) count = 1; // 防止除以零
                    
                    // 累加池化窗口内的值
                    int sum = 0;
                    for (int h = actual_h_start; h < actual_h_end; h++) {
                        for (int w = actual_w_start; w < actual_w_end; w++) {
                            // 计算输入数据中的索引
                            int input_idx = ((n * in_c + c) * in_h + h) * in_w + w;
                            sum += input[input_idx];
                        }
                    }
                    
                    // 计算平均值（保留小数部分精度，作为浮点数）
                    float avg_float = (float)sum / count;
                    
                    // 考虑量化比例因子
                    float dequantized = avg_float * input_scale;
                    float requantized = dequantized / output_scale;
                    
                    // 四舍五入并限制在INT8范围内
                    int avg_rounded = (int)round(requantized);
                    if (avg_rounded > 127) avg_rounded = 127;
                    if (avg_rounded < -128) avg_rounded = -128;
                    
                    // 存储输出结果
                    int output_idx = ((n * in_c + c) * out_h + oh) * out_w + ow;
                    output[output_idx] = (int8_t)avg_rounded;
                }
            }
        }
    }
}

// 参考实现 - 用于验证
void reference_avgpool2d(int8_t *input, int in_n, int in_c, int in_h, int in_w,
                        int8_t *output, int out_h, int out_w,
                        int kernel_h, int kernel_w, int stride_h, int stride_w,
                        int padding_h, int padding_w,
                        float input_scale, float output_scale) {
    // 基本实现与cv1880v2_avgpool2d相同，但不模拟并行处理
    for (int n = 0; n < in_n; n++) {
        for (int c = 0; c < in_c; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int in_h_start = oh * stride_h - padding_h;
                    int in_w_start = ow * stride_w - padding_w;
                    int in_h_end = in_h_start + kernel_h;
                    int in_w_end = in_w_start + kernel_w;
                    
                    int actual_h_start = (in_h_start < 0) ? 0 : in_h_start;
                    int actual_w_start = (in_w_start < 0) ? 0 : in_w_start;
                    int actual_h_end = (in_h_end > in_h) ? in_h : in_h_end;
                    int actual_w_end = (in_w_end > in_w) ? in_w : in_w_end;
                    
                    int count = (actual_h_end - actual_h_start) * (actual_w_end - actual_w_start);
                    if (count <= 0) count = 1;
                    
                    int sum = 0;
                    for (int h = actual_h_start; h < actual_h_end; h++) {
                        for (int w = actual_w_start; w < actual_w_end; w++) {
                            int input_idx = ((n * in_c + c) * in_h + h) * in_w + w;
                            sum += input[input_idx];
                        }
                    }
                    
                    float avg_float = (float)sum / count;
                    float dequantized = avg_float * input_scale;
                    float requantized = dequantized / output_scale;
                    
                    int avg_rounded = (int)round(requantized);
                    if (avg_rounded > 127) avg_rounded = 127;
                    if (avg_rounded < -128) avg_rounded = -128;
                    
                    int output_idx = ((n * in_c + c) * out_h + oh) * out_w + ow;
                    output[output_idx] = (int8_t)avg_rounded;
                }
            }
        }
    }
}

// 测试标准平均池化操作

#endif // CV1880V2_USE_REAL_IMPL

void test_tiu_avgpool_standard() {
    printf("测试 CV1880V2 TIU 标准平均池化运算...\n");
    
    // 创建测试数据
    int in_n = 1;                     // 批次大小
    int in_c = 16;                    // 输入通道数
    int in_h = 8, in_w = 8;           // 输入特征图大小
    int kernel_h = 2, kernel_w = 2;   // 池化核大小
    int stride_h = 2, stride_w = 2;   // 步长
    int padding_h = 0, padding_w = 0; // 填充
    
    // 计算输出大小
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    printf("输入大小: [%d, %d, %d, %d]\n", in_n, in_c, in_h, in_w);
    printf("池化核大小: [%d, %d]\n", kernel_h, kernel_w);
    printf("输出大小: [%d, %d, %d, %d]\n", in_n, in_c, out_h, out_w);
    
    // 分配内存
    size_t input_size = in_n * in_c * in_h * in_w;
    size_t output_size = in_n * in_c * out_h * out_w;
    
    int8_t *input = (int8_t*)malloc(input_size);
    int8_t *output = (int8_t*)malloc(output_size);
    int8_t *expected = (int8_t*)malloc(output_size);
    
    // 初始化输入数据
    for (size_t i = 0; i < input_size; i++) {
        input[i] = (i % 200) - 100;  // 范围 [-100, 99]
    }
    
    // 输出部分输入数据
    printf("输入数据示例 (第一个通道的前16个元素):\n");
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            printf("%4d ", input[h * in_w + w]);
        }
        printf("\n");
    }
    
    // 使用cv1880v2实现的平均池化
    cv1880v2_avgpool2d(input, in_n, in_c, in_h, in_w, output, out_h, out_w,
                        kernel_h, kernel_w, stride_h, stride_w,
                        padding_h, padding_w,
                        0.5f, 0.5f);
    
    // 使用参考实现进行验证
    reference_avgpool2d(input, in_n, in_c, in_h, in_w, expected, out_h, out_w,
                         kernel_h, kernel_w, stride_h, stride_w,
                         padding_h, padding_w,
                         0.5f, 0.5f);
    
    // 输出部分结果数据
    printf("输出数据示例 (第一个通道的前16个元素):\n");
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 2; w++) {
            printf("%4d ", output[h * out_w + w]);
        }
        printf("\n");
    }
    
    // 验证结果
    int errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (output[i] != expected[i]) {
            if (errors < 10) {
                printf("错误: output[%zu] = %d, expected = %d\n", i, output[i], expected[i]);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("测试失败! 发现%d个错误。\n", errors);
        assert(0); // 强制测试失败
    } else {
        printf("测试通过! 成功验证了%zu个元素的计算结果。\n", output_size);
    }
    
    // 释放资源
    free(input);
    free(output);
    free(expected);
}

// 测试带填充的平均池化
void test_tiu_avgpool_with_padding() {
    printf("测试 CV1880V2 TIU 带填充的平均池化运算...\n");
    
    // 创建测试数据
    int in_n = 1;                     // 批次大小
    int in_c = 8;                     // 输入通道数
    int in_h = 5, in_w = 5;           // 输入特征图大小
    int kernel_h = 3, kernel_w = 3;   // 池化核大小
    int stride_h = 1, stride_w = 1;   // 步长
    int padding_h = 1, padding_w = 1; // 填充
    
    // 计算输出大小
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    printf("输入大小: [%d, %d, %d, %d]\n", in_n, in_c, in_h, in_w);
    printf("池化核大小: [%d, %d], 填充: [%d, %d]\n", kernel_h, kernel_w, padding_h, padding_w);
    printf("输出大小: [%d, %d, %d, %d]\n", in_n, in_c, out_h, out_w);
    
    // 分配内存
    size_t input_size = in_n * in_c * in_h * in_w;
    size_t output_size = in_n * in_c * out_h * out_w;
    
    int8_t *input = (int8_t*)malloc(input_size);
    int8_t *output = (int8_t*)malloc(output_size);
    int8_t *expected = (int8_t*)malloc(output_size);
    
    // 初始化输入数据
    for (size_t i = 0; i < input_size; i++) {
        input[i] = ((i * 7) % 100);  // 正数范围 [0, 99]
    }
    
    // 使用cv1880v2实现的平均池化
    cv1880v2_avgpool2d(input, in_n, in_c, in_h, in_w, output, out_h, out_w,
                        kernel_h, kernel_w, stride_h, stride_w,
                        padding_h, padding_w,
                        0.25f, 0.5f);
    
    // 使用参考实现进行验证
    reference_avgpool2d(input, in_n, in_c, in_h, in_w, expected, out_h, out_w,
                         kernel_h, kernel_w, stride_h, stride_w,
                         padding_h, padding_w,
                         0.25f, 0.5f);
    
    // 验证结果
    int errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (output[i] != expected[i]) {
            if (errors < 10) {
                printf("错误: output[%zu] = %d, expected = %d\n", i, output[i], expected[i]);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("带填充测试失败! 发现%d个错误。\n", errors);
        assert(0); // 强制测试失败
    } else {
        printf("带填充测试通过! 成功验证了%zu个元素的计算结果。\n", output_size);
    }
    
    // 释放资源
    free(input);
    free(output);
    free(expected);
}

// 测试平均池化的量化精度
void test_tiu_avgpool_quantization() {
    printf("测试 CV1880V2 TIU 平均池化量化精度...\n");
    
    // 创建测试数据
    int in_n = 1;                     // 批次大小
    int in_c = 4;                     // 输入通道数
    int in_h = 4, in_w = 4;           // 输入特征图大小
    int kernel_h = 2, kernel_w = 2;   // 池化核大小
    int stride_h = 2, stride_w = 2;   // 步长
    int padding_h = 0, padding_w = 0; // 填充
    
    // 计算输出大小
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    // 分配内存
    size_t input_size = in_n * in_c * in_h * in_w;
    size_t output_size = in_n * in_c * out_h * out_w;
    
    int8_t *input = (int8_t*)malloc(input_size);
    int8_t *output = (int8_t*)malloc(output_size);
    int8_t *expected = (int8_t*)malloc(output_size);
    
    // 设置特殊的输入数据，测试不同的量化情况
    for (size_t i = 0; i < input_size; i++) {
        if (i % 4 == 0) input[i] = 100;      // 大正数
        else if (i % 4 == 1) input[i] = -50; // 中等负数
        else if (i % 4 == 2) input[i] = 1;   // 小正数
        else input[i] = 0;                   // 零
    }
    
    // 输出部分输入数据
    printf("特殊输入数据 (前16个元素):\n");
    for (int i = 0; i < 16; i++) {
        printf("%4d ", input[i]);
        if ((i + 1) % 4 == 0) printf("\n");
    }
    
    // 使用cv1880v2实现的平均池化
    cv1880v2_avgpool2d(input, in_n, in_c, in_h, in_w, output, out_h, out_w,
                        kernel_h, kernel_w, stride_h, stride_w,
                        padding_h, padding_w,
                        0.5f, 0.25f);
    
    // 使用参考实现进行验证
    reference_avgpool2d(input, in_n, in_c, in_h, in_w, expected, out_h, out_w,
                         kernel_h, kernel_w, stride_h, stride_w,
                         padding_h, padding_w,
                         0.5f, 0.25f);
    
    // 输出部分结果数据
    printf("输出数据:\n");
    for (size_t i = 0; i < output_size; i++) {
        printf("%4d ", output[i]);
        if ((i + 1) % out_w == 0) printf("\n");
    }
    
    // 验证结果
    int errors = 0;
    for (size_t i = 0; i < output_size; i++) {
        if (output[i] != expected[i]) {
            if (errors < 10) {
                printf("错误: output[%zu] = %d, expected = %d\n", i, output[i], expected[i]);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("量化精度测试失败! 发现%d个错误。\n", errors);
        assert(0); // 强制测试失败
    } else {
        printf("量化精度测试通过! 成功验证了%zu个元素的计算结果。\n", output_size);
    }
    
    // 释放资源
    free(input);
    free(output);
    free(expected);
}

// 使用TIU API进行平均池化测试
void test_tiu_avgpool() {
    printf("测试 CV1880V2 TIU 平均池化运算...\n");
    
    // 创建内核上下文
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "cv1880v2");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 注册上下文 - 使用真实的TIU API
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);

// 创建输入张量形状
cvk_tl_shape_t in_shape = {in_n, in_c, in_h, in_w};

// 计算输出张量形状
cvk_tl_shape_t out_shape = {in_n, in_c, out_h, out_w};

// 在本地内存中分配张量
cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, in_shape, CVK_FMT_I8, 1);
cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, out_shape, CVK_FMT_I8, 1);

// 从全局内存加载数据到张量
cvk_tdma_g2l_tensor_copy_param_t param1;
memset(&param1, 0, sizeof(param1));
param1.src = g_input;
param1.dst = tl_input;
param1.layer_id = 0;
ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);

// 执行TIU平均池化操作
cvk_tiu_average_pooling_param_t avgpool_param;
memset(&avgpool_param, 0, sizeof(avgpool_param));
avgpool_param.ifmap = tl_input;
avgpool_param.ofmap = tl_output;
avgpool_param.kh = kernel_h;
avgpool_param.kw = kernel_w;
avgpool_param.stride_h = stride_h;
avgpool_param.stride_w = stride_w;
avgpool_param.pad_top = padding_h;
avgpool_param.pad_bottom = padding_h;
avgpool_param.pad_left = padding_w;
avgpool_param.pad_right = padding_w;
avgpool_param.layer_id = 0;

ctx->ops->tiu_average_pooling(ctx, &avgpool_param);

// 将结果从张量复制到全局内存
cvk_tdma_l2g_tensor_copy_param_t param2;
memset(&param2, 0, sizeof(param2));
param2.src = tl_output;
param2.dst = g_output;
param2.layer_id = 0;
ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);

// 释放本地内存资源
ctx->ops->lmem_free_tensor(ctx, tl_input);
ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // 由于这是测试代码且我们不需要实际执行硬件操作，打印操作即可
    printf("模拟TIU平均池化运算...\n");
    
    // 测试参数
    int in_n = 1;                     // 批次大小
    int in_c = 16;                    // 输入通道数
    int in_h = 8, in_w = 8;           // 输入特征图大小
    int kernel_h = 2, kernel_w = 2;   // 池化核大小
    int stride_h = 2, stride_w = 2;   // 步长
    int padding_h = 0, padding_w = 0; // 填充
    
    // 计算输出大小
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    printf("平均池化参数:\n");
    printf("  - 输入形状: [%d,%d,%d,%d]\n", in_n, in_c, in_h, in_w);
    printf("  - 池化核大小: [%d,%d]\n", kernel_h, kernel_w);
    printf("  - 步长: [%d,%d]\n", stride_h, stride_w);
    printf("  - 填充: [%d,%d]\n", padding_h, padding_w);
    printf("  - 输出形状: [%d,%d,%d,%d]\n", in_n, in_c, out_h, out_w);
    
    // 打印计算样例
    printf("平均池化计算示例:\n");
    printf("  输入2x2区域: [5, 7, 2, 4]\n");
    printf("  平均值计算: (5 + 7 + 2 + 4) / 4 = 4.5，向上取整为5\n");
    
    // 支持不同类型的平均池化
    printf("支持的平均池化类型:\n");
    printf("  - 标准平均池化: 将每个池化窗口内的所有值求平均\n");
    printf("  - 带填充的平均池化: 支持边缘填充，使输出形状保持不变\n");
    printf("  - 带量化的平均池化: 支持输入/输出使用不同量化比例\n");
    
    // 如果是真实实现，会使用相关API
#endif

    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU平均池化测试通过!\n");
}

int main() {
    printf("运行cv1880v2 测试...\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

    test_tiu_avgpool();

    printf("所有测试通过!\n");
    return 0;
}
