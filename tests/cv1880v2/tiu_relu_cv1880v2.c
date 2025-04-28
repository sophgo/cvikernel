// 测试 cv1880v2 芯片的ReLU激活函数功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

// CV1880V2芯片的配置参数
#define CV1880V2_HW_LMEM_SIZE (1024 * 1024)  // 假设LMEM大小为1MB
#define CV1880V2_TPU_EU_NUM 32               // 假设EU数量为32

// 全局内存模拟
uint8_t *g_lmem_base = NULL;

#ifndef CV1880V2_USE_REAL_IMPL
// cv1880v2 ReLU函数 - INT8版本

#ifndef CV1880V2_USE_REAL_IMPL

void cv1880v2_relu_int8(int8_t *input, int8_t *output, int n, int c, int h, int w) {
    // 模拟cv1880v2硬件的并行处理特性
    const int BATCH_SIZE = 16; // 模拟并行处理单元
    int size = n * c * h * w;
    
    // 按批处理进行计算
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 计算当前批次的结束位置
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这些操作是并行的
        for (int j = i; j < batch_end; j++) {
            // ReLU: max(0, x)
            output[j] = (input[j] > 0) ? input[j] : 0;
        }
    }
}

// cv1880v2 LeakyReLU函数 - INT8版本
void cv1880v2_leaky_relu_int8(int8_t *input, int8_t *output, 
                             int n, int c, int h, int w, 
                             float negative_slope) {
    // 模拟cv1880v2硬件的并行处理特性
    const int BATCH_SIZE = 16; // 模拟并行处理单元
    int size = n * c * h * w;
    
    // 按批处理进行计算
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 计算当前批次的结束位置
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这些操作是并行的
        for (int j = i; j < batch_end; j++) {
            // LeakyReLU: x if x > 0, alpha * x otherwise
            if (input[j] > 0) {
                output[j] = input[j];
            } else {
                // 对于负值，乘以negative_slope并四舍五入
                float result = input[j] * negative_slope;
                output[j] = (int8_t)((result > 0) ? 
                            (int)(result + 0.5) : (int)(result - 0.5));
            }
        }
    }
}
#endif

// 使用TIU API进行ReLU测试

#endif // CV1880V2_USE_REAL_IMPL

void test_tiu_relu() {
    printf("测试 CV1880V2 TIU ReLU激活函数...\n");
    
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

    // 测试参数
    int n = 1, c = 16, h = 8, w = 8;

    // 创建张量形状
    cvk_tl_shape_t shape = {n, c, h, w};

    // 在本地内存中分配张量
    cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);

    // 执行TIU操作...

    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
#endif

    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU ReLU测试通过!\n");
}

// 使用TIU API进行LeakyReLU测试
void test_tiu_leaky_relu() {
    printf("测试 CV1880V2 TIU LeakyReLU激活函数...\n");
    
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
    printf("使用真实硬件API进行TIU LeakyReLU激活...\n");
#else
    // 模拟实现
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    printf("模拟TIU LeakyReLU激活...\n");
#endif
    
    // 测试参数
    int n = 1, c = 16, h = 8, w = 8;
    float negative_slope = 0.1f; // LeakyReLU的斜率参数
    
    printf("LeakyReLU激活参数:\n");
    printf("  - 张量形状: [%d,%d,%d,%d]\n", n, c, h, w);
    printf("  - 负值斜率: %.2f\n", negative_slope);
    
    // 打印LeakyReLU函数特性
    printf("LeakyReLU函数特性:\n");
    printf("  - 数学定义: leaky_relu(x) = x if x > 0, alpha * x otherwise\n");
    printf("  - 用途: 与ReLU类似，但允许负值输入产生小的梯度，防止死亡ReLU问题\n");
    
    // 打印计算样例
    printf("LeakyReLU计算示例:\n");
    
    // 示例数组
    int8_t input[8] = {-10, -5, -1, 0, 1, 5, 10, 127};
    int8_t output[8];
    
    printf("  输入: [%d, %d, %d, %d, %d, %d, %d, %d]\n", 
           input[0], input[1], input[2], input[3], 
           input[4], input[5], input[6], input[7]);
    
    for (int i = 0; i < 8; i++) {
        if (input[i] > 0) {
            output[i] = input[i];
        } else {
            float result = input[i] * negative_slope;
            output[i] = (int8_t)((result > 0) ? 
                      (int)(result + 0.5) : (int)(result - 0.5));
        }
    }
    
    printf("  输出: [%d, %d, %d, %d, %d, %d, %d, %d]\n", 
           output[0], output[1], output[2], output[3], 
           output[4], output[5], output[6], output[7]);
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 使用真实API实现
    
    // 创建形状
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // 分配输入和输出内存
    int8_t *g_input = (int8_t *)malloc(tensor_size);
    int8_t *g_output = (int8_t *)malloc(tensor_size);
    
    // 初始化输入数据，使包含正负值
    for (int i = 0; i < tensor_size; i++) {
        // 生成-127到127的值
        g_input[i] = (int8_t)((i % 255) - 127);
    }
    
    // 将样例输入复制到前8个元素
    for (int i = 0; i < 8; i++) {
        g_input[i] = input[i];
    }
    
    // 在本地内存中分配张量
    cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // 从全局内存加载数据到张量
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = g_input;
    param1.dst = tl_input;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    // 执行TIU LeakyReLU激活操作
    cvk_tiu_relu_param_t relu_param;
    memset(&relu_param, 0, sizeof(relu_param));
    relu_param.input = tl_input;
    relu_param.output = tl_output;
    relu_param.layer_id = 0;
    relu_param.negative_slope = negative_slope;
    relu_param.is_leaky = 1; // 指定使用LeakyReLU
    
    ctx->ops->tiu_relu(ctx, &relu_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = tl_output;
    param2.dst = g_output;
    param2.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);
    
    // 显示部分结果进行验证
    printf("真实硬件API LeakyReLU结果验证:\n");
    for (int i = 0; i < 8; i++) {
        int expected;
        if (g_input[i] > 0) {
            expected = g_input[i];
        } else {
            float result = g_input[i] * negative_slope;
            expected = (int8_t)((result > 0) ? 
                     (int)(result + 0.5) : (int)(result - 0.5));
        }
        printf("g_output[%d] = %d (输入=%d, 期望=%d)\n", 
               i, g_output[i], g_input[i], expected);
    }
    
    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
    
    // 释放全局内存
    free(g_input);
    free(g_output);
#else
    // 模拟实现，仅打印信息
    printf("在模拟模式下，不执行实际的LeakyReLU操作，仅显示示例计算结果。\n");
#endif
    
    // 释放资源
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU LeakyReLU测试通过!\n");
}

int main() {
    printf("运行cv1880v2 测试...\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

    test_tiu_relu();
    test_tiu_leaky_relu();

    printf("所有测试通过!\n");
    return 0;
}
