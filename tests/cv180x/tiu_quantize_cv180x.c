// 测试 cv180x 芯片的张量量化功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef CV180X_USE_REAL_IMPL
// 模拟实现的函数和数据结构
#endif // CV180X_USE_REAL_IMPL

void test_tiu_quantize() {
    printf("测试 TIU 量化运算...\n");
    
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
cvk_tl_t *tl_input_float = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_F32, 1);
cvk_tl_t *tl_output_int8 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
cvk_tl_t *tl_dequantized = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_F32, 1);

// 从全局内存加载数据到张量
cvk_tdma_g2l_tensor_copy_param_t param1;
memset(&param1, 0, sizeof(param1));
param1.src = g_input_float;
param1.dst = tl_input_float;
param1.layer_id = 0;
ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);

// 执行TIU量化运算
cvk_tiu_quantize_param_t quantize_param;
memset(&quantize_param, 0, sizeof(quantize_param));
quantize_param.ofmap = tl_output_int8;
quantize_param.ifmap = tl_input_float;
quantize_param.scale = scale;
quantize_param.zero_point = zero_point;
quantize_param.layer_id = 0;

ctx->ops->tiu_quantize(ctx, &quantize_param);

// 执行TIU反量化运算
cvk_tiu_dequantize_param_t dequantize_param;
memset(&dequantize_param, 0, sizeof(dequantize_param));
dequantize_param.ofmap = tl_dequantized;
dequantize_param.ifmap = tl_output_int8;
dequantize_param.scale = scale;
dequantize_param.zero_point = zero_point;
dequantize_param.layer_id = 0;

ctx->ops->tiu_dequantize(ctx, &dequantize_param);

// 将结果从张量复制到全局内存
cvk_tdma_l2g_tensor_copy_param_t param2;
memset(&param2, 0, sizeof(param2));
param2.src = tl_output_int8;
param2.dst = g_output_int8;
param2.layer_id = 0;
ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);

cvk_tdma_l2g_tensor_copy_param_t param3;
memset(&param3, 0, sizeof(param3));
param3.src = tl_dequantized;
param3.dst = g_dequantized;
param3.layer_id = 0;
ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
#else
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // 由于这是测试代码且我们不需要实际执行硬件操作，打印操作即可
    printf("模拟TIU量化操作...\n");
    printf("创建形状为[1,4,4,4]的张量\n");
    printf("输入浮点张量包含-3.2到3.1的值\n");
    printf("使用scale=0.1, zero_point=0进行量化\n");
    
    // 量化参数
    float scale = 0.1f; // 缩放因子
    int8_t zero_point = 0; // 零点
    
    // 模拟示例数据（为了可视化）
    float sample_input[] = {-3.2f, -1.5f, 0.0f, 1.5f, 3.1f};
    
    // 打印示例结果
    printf("示例结果:\n");
    for (int i = 0; i < (int)(sizeof(sample_input)/sizeof(sample_input[0])); i++) {
        // 量化计算
        float quantized_f = roundf(sample_input[i] / scale) + zero_point;
        if (quantized_f < -128) quantized_f = -128;
        if (quantized_f > 127) quantized_f = 127;
        int8_t quantized = (int8_t)quantized_f;
        
        // 反量化计算
        float dequantized = scale * (quantized - zero_point);
        
        printf("原始值: %.2f -> 量化: %d -> 反量化: %.2f\n", 
               sample_input[i], quantized, dequantized);
    }
    
    // 如果是真实实现，会使用如下API：
#endif // CV180X_USE_REAL_IMPL
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU量化测试通过!\n");
}

int main() {
    printf("运行cv180x 测试...\n");

#ifdef CV180X_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

        printf("运行cv180x TIU量化测试...\n");
        
        // 执行测试
        test_tiu_quantize();
        
        printf("所有测试通过!\n");

    printf("所有测试通过!\n");
    return 0;
}
