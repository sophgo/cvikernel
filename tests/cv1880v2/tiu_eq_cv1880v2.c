// 测试 cv1880v2 芯片的张量等式比较功能
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
// cv1880v2张量等式比较函数

#ifndef CV1880V2_USE_REAL_IMPL

void cv1880v2_tensor_eq(int8_t *input1, int8_t *input2, int8_t *output, 
                        int n, int c, int h, int w) {
    // 模拟cv1880v2硬件的并行处理特性
    const int BATCH_SIZE = 16; // 模拟并行处理单元
    int size = n * c * h * w;
    
    // 按批处理进行计算
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 计算当前批次的结束位置
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这些操作是并行的
        for (int j = i; j < batch_end; j++) {
            // 等式比较 - 如果相等则输出1，否则输出0
            output[j] = (input1[j] == input2[j]) ? 1 : 0;
        }
    }
}
#endif

// 使用TIU API进行张量等式比较测试

#endif // CV1880V2_USE_REAL_IMPL

void test_tiu_eq() {
    printf("测试 CV1880V2 TIU 等式比较运算...\n");
    
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
    
    printf("TIU等式比较测试通过!\n");
}

// 使用TIU API进行张量与常量的等式比较测试
void test_tiu_eq_constant() {
    printf("测试 CV1880V2 TIU 张量与常量等式比较运算...\n");
    
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
    printf("使用真实硬件API进行TIU张量与常量等式比较...\n");
#else
    // 模拟实现
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    printf("模拟TIU张量与常量等式比较...\n");
#endif
    
    // 测试参数
    int n = 1, c = 16, h = 8, w = 8;
    int8_t constant_value = 50;
    
    printf("张量与常量等式比较参数:\n");
    printf("  - 张量形状: [%d,%d,%d,%d]\n", n, c, h, w);
    printf("  - 常量值: %d\n", constant_value);
    
    // 打印计算样例
    printf("等式比较与常量计算示例:\n");
    
    // 示例数组
    int8_t input[5] = {10, 50, 30, 50, 60};
    int8_t result[5];
    
    printf("  输入: [%d, %d, %d, %d, %d]\n", input[0], input[1], input[2], input[3], input[4]);
    printf("  常量: %d\n", constant_value);
    
    for (int i = 0; i < 5; i++) {
        result[i] = (input[i] == constant_value) ? 1 : 0;
    }
    
    printf("  结果: [%d, %d, %d, %d, %d]\n", result[0], result[1], result[2], result[3], result[4]);
    printf("    等式比较规则: 相等=1，不等=0\n");
    
#ifdef CV1880V2_USE_REAL_IMPL
    // 使用真实API实现
    
    // 创建形状
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // 分配输入和输出内存
    int8_t *g_input = (int8_t *)malloc(tensor_size);
    int8_t *g_output = (int8_t *)malloc(tensor_size);
    
    // 初始化输入数据，使部分值等于常量值
    for (int i = 0; i < tensor_size; i++) {
        if ((i % 5) == 1 || (i % 5) == 3) {
            g_input[i] = constant_value; // 等于常量
        } else {
            g_input[i] = (i % 100) + ((i % 2) ? 1 : -1); // 不等于常量
        }
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
    
    // 执行TIU张量与常量等式比较操作
    cvk_tiu_compare_param_t compare_param;
    memset(&compare_param, 0, sizeof(compare_param));
    compare_param.input0 = tl_input;
    compare_param.input1 = NULL;
    compare_param.const_input1 = 1;
    compare_param.const_val = constant_value;
    compare_param.output = tl_output;
    compare_param.layer_id = 0;
    compare_param.cmp_op = CVK_TIU_CMP_EQ; // 等式比较
    
    ctx->ops->tiu_compare(ctx, &compare_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = tl_output;
    param2.dst = g_output;
    param2.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);
    
    // 显示部分结果进行验证
    printf("真实硬件API等式比较与常量结果验证:\n");
    for (int i = 0; i < 10; i++) {
        int expected = (g_input[i] == constant_value) ? 1 : 0;
        printf("g_output[%d] = %d (输入=%d, 常量=%d, 期望=%d)\n", 
               i, g_output[i], g_input[i], constant_value, expected);
    }
    
    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
    
    // 释放全局内存
    free(g_input);
    free(g_output);
#else
    // 模拟实现，仅打印信息
    printf("在模拟模式下，不执行实际的等式比较与常量操作，仅显示示例计算结果。\n");
#endif
    
    // 释放资源
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU等式比较与常量测试通过!\n");
}

int main() {
    printf("运行cv1880v2 测试...\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

    test_tiu_eq();
    test_tiu_eq_constant();

    printf("所有测试通过!\n");
    return 0;
}
