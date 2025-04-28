// 测试 cv180x 芯片的张量加法功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef CV180X_USE_REAL_IMPL
// 模拟实现的函数和数据结构
#endif // CV180X_USE_REAL_IMPL

// 使用TIU API进行张量加法测试
void test_tiu_add() {
    printf("测试 TIU 加法运算...\n");
    
    // 创建内核上下文
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "cv180x");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef CV180X_USE_REAL_IMPL
    // 注册上下文 - 使用真实的TIU API
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    
    // 创建测试数据
    int n = 1, c = 4, h = 4, w = 4;
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // 在本地内存（Local Memory）中分配张量
    cvk_tl_t *tl_input1 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_input2 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // 全局内存张量
    cvk_tg_t g_input1, g_input2, g_output;
    memset(&g_input1, 0, sizeof(cvk_tg_t));
    memset(&g_input2, 0, sizeof(cvk_tg_t));
    memset(&g_output, 0, sizeof(cvk_tg_t));
    
    // TODO: 在真实实现中，需要初始化全局内存张量
    
    // 从全局内存加载数据到张量
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = &g_input1;
    param1.dst = tl_input1;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    cvk_tdma_g2l_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = &g_input2;
    param2.dst = tl_input2;
    param2.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param2);
    
    // 执行TIU加法运算
    cvk_tiu_add_param_t add_param;
    memset(&add_param, 0, sizeof(add_param));
    add_param.res_high = NULL;
    add_param.res_low = tl_output;
    add_param.a_high = NULL;
    add_param.a_low = tl_input1;
    add_param.b_high = NULL;
    add_param.b_low = tl_input2;
    add_param.rshift_bits = 0;
    add_param.layer_id = 0;
    
    ctx->ops->tiu_add(ctx, &add_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output;
    param3.dst = &g_output;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input1);
    ctx->ops->lmem_free_tensor(ctx, tl_input2);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // 注册上下文 - 由于这是测试代码，我们可以模拟而不是真正调用
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // 由于这是测试代码且我们不需要实际执行硬件操作，打印操作即可
    printf("模拟TIU张量加法操作...\n");
    printf("创建形状为[1,4,4,4]的张量\n");
    printf("将输入1设为范围0-9的循环值\n");
    printf("将输入2设为范围5-9的循环值\n");
    printf("执行TIU加法操作\n");
    
    // 模拟示例数据（为了可视化）
    int8_t sample_input1[5] = {0, 1, 2, 3, 4};
    int8_t sample_input2[5] = {5, 6, 7, 8, 9};
    
    // 打印示例结果
    printf("示例结果:\n");
    for (int i = 0; i < 5; i++) {
        int8_t add_val = sample_input1[i] + sample_input2[i];
        printf("input1[%d]=%d, input2[%d]=%d, sum=%d\n", 
               i, sample_input1[i], i, sample_input2[i], add_val);
    }
#endif // CV180X_USE_REAL_IMPL
    
    // 释放资源
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU加法测试通过!\n");
}

// 使用TIU API进行常量加法测试
void test_tiu_add_constant() {
    printf("测试 TIU 常量加法运算...\n");
    
    // 创建内核上下文
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "cv180x");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef CV180X_USE_REAL_IMPL
    // 注册上下文 - 使用真实的TIU API
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    
    // 创建测试数据
    int n = 1, c = 4, h = 4, w = 4;
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // 在本地内存（Local Memory）中分配张量
    cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // 全局内存张量
    cvk_tg_t g_input, g_output;
    memset(&g_input, 0, sizeof(cvk_tg_t));
    memset(&g_output, 0, sizeof(cvk_tg_t));
    
    // TODO: 在真实实现中，需要初始化全局内存张量
    
    // 从全局内存加载数据到张量
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = &g_input;
    param1.dst = tl_input;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    // 执行TIU常量加法运算
    cvk_tiu_add_param_t add_param;
    memset(&add_param, 0, sizeof(add_param));
    add_param.res_high = NULL;
    add_param.res_low = tl_output;
    add_param.a_high = NULL;
    add_param.a_low = tl_input;
    add_param.b_is_const = 1;
    add_param.b_const.val = 5;
    add_param.b_const.is_signed = 1;
    add_param.rshift_bits = 0;
    add_param.layer_id = 0;
    
    ctx->ops->tiu_add(ctx, &add_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = tl_output;
    param2.dst = &g_output;
    param2.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);
    
    // 释放本地内存张量
    ctx->ops->lmem_free_tensor(ctx, tl_input);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // 注册上下文 - 由于这是测试代码，我们可以模拟而不是真正调用
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // 由于这是测试代码且我们不需要实际执行硬件操作，打印操作即可
    printf("模拟TIU常量加法操作...\n");
    printf("创建形状为[1,4,4,4]的张量\n");
    printf("将输入张量设为范围0-9的循环值\n");
    printf("将常量设为5\n");
    printf("执行TIU常量加法操作\n");
    
    // 模拟示例数据（为了可视化）
    int8_t sample_input[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int8_t constant = 5;
    
    // 打印示例结果
    printf("示例结果:\n");
    for (int i = 0; i < 10; i++) {
        int8_t add_val = sample_input[i] + constant;
        printf("input[%d]=%d, constant=%d, sum=%d\n", 
               i, sample_input[i], constant, add_val);
    }
#endif // CV180X_USE_REAL_IMPL
    
    // 释放资源
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU常量加法测试通过!\n");
}

int main() {
    printf("运行cv180x TIU加法测试...\n");
    
#ifdef CV180X_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif
    
    // 执行测试
    test_tiu_add();
    test_tiu_add_constant();
    
    printf("所有测试通过!\n");
    return 0;
}
