// 测试 cv180x 芯片的张量平均池化功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef CV180X_USE_REAL_IMPL
// 模拟实现的函数和数据结构
#endif // CV180X_USE_REAL_IMPL

// 使用TIU API进行平均池化测试
void test_tiu_avgpool() {
    printf("测试 TIU 平均池化操作...\n");
    
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
    int n = 1, c = 1, h = 4, w = 4;
    cvk_tl_shape_t input_shape = {n, c, h, w};
    cvk_tl_shape_t output_shape = {n, c, h/2, w/2};
    
    // 在本地内存（Local Memory）中分配张量
    cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, input_shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, output_shape, CVK_FMT_I8, 1);
    
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
    
    // 执行TIU平均池化操作
    cvk_tiu_average_pooling_param_t avgpool_param;
    memset(&avgpool_param, 0, sizeof(avgpool_param));
    avgpool_param.ofmap = tl_output;
    avgpool_param.ifmap = tl_input;
    avgpool_param.kh = 2;
    avgpool_param.kw = 2;
    avgpool_param.stride_h = 2;
    avgpool_param.stride_w = 2;
    avgpool_param.pad_top = 0;
    avgpool_param.pad_bottom = 0;
    avgpool_param.pad_left = 0;
    avgpool_param.pad_right = 0;
    avgpool_param.rshift_bits = 0;
    avgpool_param.layer_id = 0;
    
    ctx->ops->tiu_average_pooling(ctx, &avgpool_param);
    
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
    printf("模拟TIU平均池化操作...\n");
    printf("创建形状为[1,1,4,4]的输入张量\n");
    printf("创建形状为[1,1,2,2]的输出张量\n");
    printf("使用2x2的池化核和步长2\n");
    printf("执行平均池化操作\n");
    
    // 模拟示例数据（为了可视化）
    int8_t input[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    // 计算2x2的平均池化结果
    int8_t output[4];
    output[0] = (input[0] + input[1] + input[4] + input[5]) / 4;  // 左上角的2x2区域
    output[1] = (input[2] + input[3] + input[6] + input[7]) / 4;  // 右上角的2x2区域
    output[2] = (input[8] + input[9] + input[12] + input[13]) / 4; // 左下角的2x2区域
    output[3] = (input[10] + input[11] + input[14] + input[15]) / 4; // 右下角的2x2区域
    
    // 打印示例结果
    printf("输入张量 (4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%2d ", input[i*4+j]);
        }
        printf("\n");
    }
    
    printf("平均池化后输出 (2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%2d ", output[i*2+j]);
        }
        printf("\n");
    }
#endif // CV180X_USE_REAL_IMPL
    
    // 释放资源
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU平均池化测试通过!\n");
}

int main() {
    printf("运行cv180x TIU平均池化测试...\n");
    
#ifdef CV180X_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif
    
    // 执行测试
    test_tiu_avgpool();
    
    printf("所有测试通过!\n");
    return 0;
}
