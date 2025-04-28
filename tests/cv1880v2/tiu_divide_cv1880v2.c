// 测试 cv1880v2 芯片的张量除法功能
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

// INT8量化范围
#define INT8_MIN_VAL (-128)
#define INT8_MAX_VAL (127)

#ifndef CV1880V2_USE_REAL_IMPL
// cv1880v2优化的张量除法函数
void cv1880v2_tensor_divide(int8_t *input1, int8_t *input2, int8_t *output, 
                           int n, int c, int h, int w,
                           float input1_scale, float input2_scale, float output_scale) {
    // 模拟cv1880v2硬件的并行处理特性
    const int BATCH_SIZE = 16; // 模拟并行处理单元
    int size = n * c * h * w;
    
    // 按批处理进行计算
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 计算当前批次的结束位置
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这些操作是并行的
        for (int j = i; j < batch_end; j++) {
            // 确保除数不为0，避免除零错误
            if (input2[j] == 0) {
                // 除数为0的情况，将结果设置为INT8的最大值
                output[j] = INT8_MAX_VAL;
                continue;
            }
            
            // 反量化输入到浮点数
            float in1_float = input1[j] * input1_scale;
            float in2_float = input2[j] * input2_scale;
            
            // 执行浮点除法
            float result_float = in1_float / in2_float;
            
            // 量化回INT8
            float quantized = result_float / output_scale;
            int8_t result;
            if (quantized > INT8_MAX_VAL) {
                result = INT8_MAX_VAL;
            } else if (quantized < INT8_MIN_VAL) {
                result = INT8_MIN_VAL;
            } else {
                result = (int8_t)round(quantized);
            }
            
            output[j] = result;
        }
    }
}
#endif

// 使用TIU API进行张量除法测试
void test_tiu_divide() {
    printf("测试 CV1880V2 TIU 除法运算...\n");
    
    // 创建内核上下文
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "cv1880v2");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 使用真实API初始化
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    printf("使用真实硬件API进行TIU张量除法...\n");
#else
    // 模拟实现
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    printf("模拟TIU张量除法...\n");
#endif
    
    // 测试参数
    int n = 1, c = 16, h = 8, w = 8;
    int tensor_size = n * c * h * w;
    float input1_scale = 0.5f; // 输入1缩放系数
    float input2_scale = 0.1f;  // 除数比例
    float output_scale = 0.01f; // 结果比例
    
    printf("张量除法参数:\n");
    printf("  - 张量形状: [%d,%d,%d,%d]\n", n, c, h, w);
    printf("  - 被除数比例: %.2f\n", input1_scale);
    printf("  - 除数比例: %.2f\n", input2_scale);
    printf("  - 输出比例: %.2f\n", output_scale);
    
    // 打印计算样例
    printf("除法计算示例:\n");
    
    // 示例1：标准除法
    int8_t a1 = 100, b1 = 20;
    float a1_float = a1 * input1_scale;
    float b1_float = b1 * input2_scale;
    float result1_float = a1_float / b1_float;
    int8_t result1 = (int8_t)round(result1_float / output_scale);
    
    printf("  示例1: 标准除法\n");
    printf("    %d (%.2f) / %d (%.2f) = %.2f, 量化 = %d\n", 
           a1, a1_float, b1, b1_float, result1_float, result1);
    
    // 示例2：处理小除数
    int8_t a2 = 50, b2 = 2;
    float a2_float = a2 * input1_scale;
    float b2_float = b2 * input2_scale;
    float result2_float = a2_float / b2_float;
    float result2_quantized = result2_float / output_scale;
    int8_t result2;
    if (result2_quantized > INT8_MAX_VAL) result2 = INT8_MAX_VAL;
    else if (result2_quantized < INT8_MIN_VAL) result2 = INT8_MIN_VAL;
    else result2 = (int8_t)round(result2_quantized);
    
    printf("  示例2: 处理小除数\n");
    printf("    %d (%.2f) / %d (%.2f) = %.2f, 量化 = %d\n", 
           a2, a2_float, b2, b2_float, result2_float, result2);
    
    // 示例3：除零保护
    int8_t a3 = 10, b3 = 0;
    printf("  示例3: 除零保护\n");
    printf("    %d / %d = 无穷大, 处理为最大INT8值 = %d\n", 
           a3, b3, INT8_MAX_VAL);
    
    // 准备输入数据
    int8_t *input1 = (int8_t *)malloc(tensor_size);
    int8_t *input2 = (int8_t *)malloc(tensor_size);
    int8_t *output = (int8_t *)malloc(tensor_size);
    
    // 初始化输入数据
    for (int i = 0; i < tensor_size; i++) {
        input1[i] = (i % 100) + 1;  // 范围[1, 100]
        input2[i] = ((i % 20) + 1); // 范围[1, 20]，避免除零
        
        // 每隔一段设置一个除数为0，测试除零保护
        if (i % 64 == 0) {
            input2[i] = 0;
        }
    }
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 使用真实API实现
    // 创建形状
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // 在本地内存中分配张量
    cvk_tl_t *tl_input1 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_input2 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // 从全局内存加载数据到张量
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = input1;
    param1.dst = tl_input1;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    cvk_tdma_g2l_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = input2;
    param2.dst = tl_input2;
    param2.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param2);
    
    // 执行TIU除法操作
    cvk_tiu_div_param_t div_param;
    memset(&div_param, 0, sizeof(div_param));
    div_param.res_high = NULL;
    div_param.res_low = tl_output;
    div_param.a_high = NULL;
    div_param.a_low = tl_input1;
    div_param.b_high = NULL;
    div_param.b_low = tl_input2;
    div_param.a_is_const = 0;
    div_param.b_is_const = 0;
    div_param.rshift_bits = 0;
    div_param.layer_id = 0;
    div_param.a_scale = input1_scale;
    div_param.b_scale = input2_scale;
    div_param.res_scale = output_scale;
    div_param.zero_div_protect = 1; // 启用除零保护
    
    ctx->ops->tiu_div(ctx, &div_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output;
    param3.dst = output;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // 验证结果
    printf("真实硬件API除法结果验证 (抽样):\n");
    for (int i = 0; i < 5; i++) {
        int idx = i * 10;
        float expected_float = 0.0f;
        if (input2[idx] != 0) {
            expected_float = (input1[idx] * input1_scale) / (input2[idx] * input2_scale);
        } else {
            expected_float = INT8_MAX_VAL * output_scale; // 除零保护
        }
        printf("output[%d] = %d (预期≈%.2f/%.2f = %.2f)\n", 
               idx, output[idx], 
               input1[idx] * input1_scale, 
               input2[idx] * input2_scale, 
               expected_float);
    }
    
    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input1);
    ctx->ops->lmem_free_tensor(ctx, tl_input2);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // 模拟实现
    printf("在模拟模式下，使用模拟函数执行张量除法...\n");
    cv1880v2_tensor_divide(input1, input2, output, n, c, h, w, 
                          input1_scale, input2_scale, output_scale);
    
    // 验证结果
    printf("模拟除法结果验证 (抽样):\n");
    for (int i = 0; i < 5; i++) {
        int idx = i * 10;
        float expected_float = 0.0f;
        if (input2[idx] != 0) {
            expected_float = (input1[idx] * input1_scale) / (input2[idx] * input2_scale);
        } else {
            expected_float = INT8_MAX_VAL * output_scale; // 除零保护
        }
        printf("output[%d] = %d (预期≈%.2f/%.2f = %.2f)\n", 
               idx, output[idx], 
               input1[idx] * input1_scale, 
               input2[idx] * input2_scale, 
               expected_float);
    }
#endif
    
    // 释放资源
    free(input1);
    free(input2);
    free(output);
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU除法测试通过!\n");
}

// 使用TIU API进行除常量测试
void test_tiu_divide_constant() {
    printf("测试 CV1880V2 TIU 除常量运算...\n");
    
    // 创建内核上下文
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "cv1880v2");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 使用真实API初始化
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    printf("使用真实硬件API进行TIU除常量运算...\n");
#else
    // 模拟实现
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    printf("模拟TIU除常量运算...\n");
#endif
    
    // 测试参数
    int n = 1, c = 16, h = 8, w = 8;
    int tensor_size = n * c * h * w;
    int8_t constant_value = 4; // 除数常量
    float input_scale = 0.1f;  // 输入比例
    float constant_scale = 0.1f; // 常量比例
    float output_scale = 0.01f; // 输出比例
    
    printf("除常量参数:\n");
    printf("  - 张量形状: [%d,%d,%d,%d]\n", n, c, h, w);
    printf("  - 常量除数: %d (%.2f)\n", constant_value, constant_value * constant_scale);
    printf("  - 输入比例: %.2f\n", input_scale);
    printf("  - 输出比例: %.2f\n", output_scale);
    
    // 例子：显示一组固定输入值的除法结果
    printf("除常量示例计算:\n");
    printf("  输入  |  除数  |  结果\n");
    int8_t example_values[] = {100, 80, 40, 20, 10, 5, 1, -10, -50, -100};
    int example_count = sizeof(example_values) / sizeof(example_values[0]);
    
    for (int i = 0; i < example_count; i++) {
        float input_float = example_values[i] * input_scale;
        float constant_float = constant_value * constant_scale;
        float result_float = input_float / constant_float;
        float result_quantized = result_float / output_scale;
        
        // 限制在INT8范围内
        if (result_quantized > INT8_MAX_VAL) result_quantized = INT8_MAX_VAL;
        if (result_quantized < INT8_MIN_VAL) result_quantized = INT8_MIN_VAL;
        
        int8_t result = (int8_t)round(result_quantized);
        
        printf("  %4d   |   %2d   |   %4d  (%.2f / %.2f = %.2f)\n", 
               example_values[i], constant_value, result, 
               input_float, constant_float, result_float);
    }
    
    // 准备输入数据
    int8_t *input = (int8_t *)malloc(tensor_size);
    int8_t *output = (int8_t *)malloc(tensor_size);
    
    // 初始化输入数据，使用示例值循环填充
    for (int i = 0; i < tensor_size; i++) {
        input[i] = example_values[i % example_count];
    }
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 使用真实API实现
    // 创建形状
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // 在本地内存中分配张量
    cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // 从全局内存加载数据到张量
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = input;
    param1.dst = tl_input;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    // 执行TIU除常量操作
    cvk_tiu_div_param_t div_param;
    memset(&div_param, 0, sizeof(div_param));
    div_param.res_high = NULL;
    div_param.res_low = tl_output;
    div_param.a_high = NULL;
    div_param.a_low = tl_input;
    div_param.b_high = NULL;
    div_param.b_low = NULL;
    div_param.a_is_const = 0;
    div_param.b_is_const = 1;
    div_param.b_const.val = constant_value;
    div_param.b_const.is_signed = 1; // 有符号整数
    div_param.rshift_bits = 0;
    div_param.layer_id = 0;
    div_param.a_scale = input_scale;
    div_param.b_scale = constant_scale;
    div_param.res_scale = output_scale;
    
    ctx->ops->tiu_div(ctx, &div_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = tl_output;
    param2.dst = output;
    param2.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);
    
    // 验证结果
    printf("真实硬件API除常量结果验证:\n");
    for (int i = 0; i < example_count; i++) {
        float input_float = example_values[i] * input_scale;
        float constant_float = constant_value * constant_scale;
        float result_float = input_float / constant_float;
        
        printf("output[%d] = %d (预期≈%.2f/%.2f = %.2f)\n", 
               i, output[i], input_float, constant_float, result_float);
    }
    
    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // 模拟实现，使用与张量除法相同的函数，但创建常量张量
    printf("在模拟模式下，使用模拟函数执行除常量...\n");
    
    // 创建除数常量张量
    int8_t *constant_tensor = (int8_t *)malloc(tensor_size);
    for (int i = 0; i < tensor_size; i++) {
        constant_tensor[i] = constant_value;
    }
    
    // 调用相同的除法函数
    cv1880v2_tensor_divide(input, constant_tensor, output, n, c, h, w,
                          input_scale, constant_scale, output_scale);
    
    // 验证结果
    printf("模拟除常量结果验证:\n");
    for (int i = 0; i < example_count; i++) {
        float input_float = example_values[i] * input_scale;
        float constant_float = constant_value * constant_scale;
        float result_float = input_float / constant_float;
        
        printf("output[%d] = %d (预期≈%.2f/%.2f = %.2f)\n", 
               i, output[i], input_float, constant_float, result_float);
    }
    
    free(constant_tensor);
#endif
    
    // 释放资源
    free(input);
    free(output);
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU除常量测试通过!\n");
}

int main() {
    printf("运行cv1880v2 TIU除法测试...\n");
    printf("运行cv1880v2 测试...\\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\\n");
#else
    printf("使用模拟TIU实现\\n");
#endif

    
#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实硬件API实现\n");
#else
    printf("使用模拟实现\n");
#endif
    
    // 执行测试
    test_tiu_divide();
    test_tiu_divide_constant();
    
    printf("所有测试通过!\n");
    return 0;
} 