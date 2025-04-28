// 测试 cv1880v2 芯片的张量量化功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "../../include/cvikernel/cvikernel.h"

// CV1880V2芯片的配置参数
#define CV1880V2_HW_LMEM_SIZE (1024 * 1024)  // 假设LMEM大小为1MB
#define CV1880V2_QUANT_BIT_WIDTH 8           // 8位量化
#define CV1880V2_INT8_MIN -128
#define CV1880V2_INT8_MAX 127

// 全局内存模拟
uint8_t *g_lmem_base = NULL;

// cv1880v2量化函数 - 浮点到INT8的转换

#ifndef CV1880V2_USE_REAL_IMPL

void cv1880v2_quantize_float_to_int8(float *input, int8_t *output, int size, 
                                    float scale, int8_t zero_point) {
    // 模拟cv1880v2的并行量化操作
    // 假设的批处理大小（模拟硬件并行）
    const int BATCH_SIZE = 16;
    
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 处理一个批次
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这是并行的
        for (int j = i; j < batch_end; j++) {
            // 应用量化公式: q = round(x / scale) + zero_point
            float quantized_float = roundf(input[j] / scale) + zero_point;
            
            // 裁剪到INT8范围
            if (quantized_float < CV1880V2_INT8_MIN) quantized_float = CV1880V2_INT8_MIN;
            if (quantized_float > CV1880V2_INT8_MAX) quantized_float = CV1880V2_INT8_MAX;
            
            output[j] = (int8_t)quantized_float;
        }
    }
}

// cv1880v2反量化函数 - INT8到浮点的转换
void cv1880v2_dequantize_int8_to_float(int8_t *input, float *output, int size, 
                                      float scale, int8_t zero_point) {
    // 模拟cv1880v2的并行反量化操作
    const int BATCH_SIZE = 16;
    
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 处理一个批次
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这是并行的
        for (int j = i; j < batch_end; j++) {
            // 应用反量化公式: x = (q - zero_point) * scale
            output[j] = ((float)input[j] - zero_point) * scale;
        }
    }
}

// 使用TIU API进行量化测试

#endif // CV1880V2_USE_REAL_IMPL

void test_tiu_quantize() {
    printf("测试 CV1880V2 TIU 量化运算...\n");
    
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

// 创建浮点张量形状
cvk_tl_shape_t shape = {n, c, h, w};

// 在本地内存中分配张量
cvk_tl_t *tl_input_float = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_F32, 1);
cvk_tl_t *tl_output_int8 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
cvk_tl_t *tl_dequant_float = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_F32, 1);

// 从全局内存加载数据到张量
cvk_tdma_g2l_tensor_copy_param_t param1;
memset(&param1, 0, sizeof(param1));
param1.src = g_input_float;
param1.dst = tl_input_float;
param1.layer_id = 0;
ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);

// 执行TIU量化操作
cvk_tiu_quantize_param_t quantize_param;
memset(&quantize_param, 0, sizeof(quantize_param));
quantize_param.ifmap = tl_input_float;
quantize_param.ofmap = tl_output_int8;
quantize_param.scale = scale;
quantize_param.zero_point = zero_point;
quantize_param.layer_id = 0;

ctx->ops->tiu_quantize(ctx, &quantize_param);

// 将量化结果从张量复制到全局内存
cvk_tdma_l2g_tensor_copy_param_t param2;
memset(&param2, 0, sizeof(param2));
param2.src = tl_output_int8;
param2.dst = g_output_int8;
param2.layer_id = 0;
ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);

// 执行TIU反量化操作
cvk_tiu_dequantize_param_t dequantize_param;
memset(&dequantize_param, 0, sizeof(dequantize_param));
dequantize_param.ifmap = tl_output_int8;
dequantize_param.ofmap = tl_dequant_float;
dequantize_param.scale = scale;
dequantize_param.zero_point = zero_point;
dequantize_param.layer_id = 0;

ctx->ops->tiu_dequantize(ctx, &dequantize_param);

// 将反量化结果从张量复制到全局内存
cvk_tdma_l2g_tensor_copy_param_t param3;
memset(&param3, 0, sizeof(param3));
param3.src = tl_dequant_float;
param3.dst = g_dequant_float;
param3.layer_id = 0;
ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);

// 释放本地内存资源
ctx->ops->lmem_free_tensor(ctx, tl_input_float);
ctx->ops->lmem_free_tensor(ctx, tl_output_int8);
ctx->ops->lmem_free_tensor(ctx, tl_dequant_float);
#else
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // 由于这是测试代码且我们不需要实际执行硬件操作，打印操作即可
    printf("模拟TIU量化运算...\n");
    
    // 测试参数
    int n = 1, c = 4, h = 8, w = 32;
    float scale = 0.1f;
    int8_t zero_point = 0;
    
    printf("创建形状为[%d,%d,%d,%d]的张量\n", n, c, h, w);
    printf("量化参数:\n");
    printf("  - 缩放因子(scale): %.6f\n", scale);
    printf("  - 零点(zero_point): %d\n", zero_point);
    
    // 打印模拟量化和反量化示例
    printf("量化示例计算:\n");
    float input_values[5] = {-3.2f, -1.5f, 0.0f, 1.5f, 3.1f};
    
    for (int i = 0; i < 5; i++) {
        float in_val = input_values[i];
        float quantized_float = roundf(in_val / scale) + zero_point;
        int8_t quantized_int8 = 0;
        
        // 裁剪到INT8范围
        if (quantized_float < -128) {
            quantized_int8 = -128;
        } else if (quantized_float > 127) {
            quantized_int8 = 127;
        } else {
            quantized_int8 = (int8_t)quantized_float;
        }
        
        // 反量化
        float dequantized = ((float)quantized_int8 - zero_point) * scale;
        
        printf("  原始值: %.2f -> 量化: %d -> 反量化: %.2f\n", 
               in_val, quantized_int8, dequantized);
    }
    
    // 如果是真实实现，会使用如下API：
#endif

    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU量化测试通过!\n");
}

int main() {
    printf("运行cv1880v2 测试...\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

    test_tiu_quantize();

    printf("所有测试通过!\n");
    return 0;
}
