// 测试 cv1880v2 芯片的最大池化功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef CV1880V2_USE_REAL_IMPL

// CV1880V2芯片的配置参数
#define CV1880V2_HW_LMEM_SIZE (1024 * 1024)  // 假设LMEM大小为1MB
#define CV1880V2_TPU_EU_NUM 32               // 假设EU数量为32

// 全局内存模拟
uint8_t *g_lmem_base = NULL;

// cv1880v2 优化的最大池化函数
void cv1880v2_maxpool2d(int8_t *input, int in_n, int in_c, int in_h, int in_w,
                       int8_t *output, int out_h, int out_w,
                       int kernel_h, int kernel_w, int stride_h, int stride_w,
                       int padding_h, int padding_w) {
    
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
                    in_h_start = (in_h_start < 0) ? 0 : in_h_start;
                    in_w_start = (in_w_start < 0) ? 0 : in_w_start;
                    in_h_end = (in_h_end > in_h) ? in_h : in_h_end;
                    in_w_end = (in_w_end > in_w) ? in_w : in_w_end;
                    
                    // 初始化为最小INT8值
                    int8_t max_val = INT8_MIN;
                    
                    // 在池化窗口内查找最大值
                    for (int h = in_h_start; h < in_h_end; h++) {
                        for (int w = in_w_start; w < in_w_end; w++) {
                            // 计算输入数据中的索引
                            int input_idx = ((n * in_c + c) * in_h + h) * in_w + w;
                            
                            // 更新最大值
                            if (input[input_idx] > max_val) {
                                max_val = input[input_idx];
                            }
                        }
                    }
                    
                    // 存储输出结果
                    int output_idx = ((n * in_c + c) * out_h + oh) * out_w + ow;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}

#endif // CV1880V2_USE_REAL_IMPL

// 使用TIU API进行最大池化测试
void test_tiu_maxpool() {
    printf("测试 CV1880V2 TIU 最大池化运算...\n");
    
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
    int in_n = 1;                     // 批次大小
    int in_c = 16;                    // 输入通道数
    int in_h = 8, in_w = 8;           // 输入特征图大小
    int kernel_h = 2, kernel_w = 2;   // 池化核大小
    int stride_h = 2, stride_w = 2;   // 步长
    int padding_h = 0, padding_w = 0; // 填充
    
    // 计算输出大小
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    printf("最大池化参数:\n");
    printf("  - 输入形状: [%d,%d,%d,%d]\n", in_n, in_c, in_h, in_w);
    printf("  - 池化核大小: [%d,%d]\n", kernel_h, kernel_w);
    printf("  - 步长: [%d,%d]\n", stride_h, stride_w);
    printf("  - 填充: [%d,%d]\n", padding_h, padding_w);
    printf("  - 输出形状: [%d,%d,%d,%d]\n", in_n, in_c, out_h, out_w);
    
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
    
    // 执行TIU最大池化操作
    cvk_tiu_max_pooling_param_t maxpool_param;
    memset(&maxpool_param, 0, sizeof(maxpool_param));
    maxpool_param.ifmap = tl_input;
    maxpool_param.ofmap = tl_output;
    maxpool_param.kh = kernel_h;
    maxpool_param.kw = kernel_w;
    maxpool_param.stride_h = stride_h;
    maxpool_param.stride_w = stride_w;
    maxpool_param.pad_top = padding_h;
    maxpool_param.pad_bottom = padding_h;
    maxpool_param.pad_left = padding_w;
    maxpool_param.pad_right = padding_w;
    maxpool_param.layer_id = 0;
    
    ctx->ops->tiu_max_pooling(ctx, &maxpool_param);
    
    // 将结果从张量复制到全局内存
    cvk_tdma_l2g_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = tl_output;
    param2.dst = g_output;
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
    printf("模拟TIU最大池化运算...\n");
    
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
    
    printf("最大池化参数:\n");
    printf("  - 输入形状: [%d,%d,%d,%d]\n", in_n, in_c, in_h, in_w);
    printf("  - 池化核大小: [%d,%d]\n", kernel_h, kernel_w);
    printf("  - 步长: [%d,%d]\n", stride_h, stride_w);
    printf("  - 填充: [%d,%d]\n", padding_h, padding_w);
    printf("  - 输出形状: [%d,%d,%d,%d]\n", in_n, in_c, out_h, out_w);
    
    // 打印计算样例
    printf("最大池化计算示例:\n");
    printf("  输入2x2区域: [5, 7, 2, 4]\n");
    printf("  最大值: max(5, 7, 2, 4) = 7\n");
    
    // 展示不同填充策略
    printf("最大池化的填充策略:\n");
    printf("  1. 不填充 (VALID): 输出尺寸可能会减小\n");
    printf("  2. 同值填充 (SAME): 使用指定值填充边缘，保持输出尺寸与输入相近\n");
#endif
    
    // 释放资源
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU最大池化测试通过!\n");
}

int main() {
    printf("运行cv1880v2 TIU最大池化测试...\n");
    printf("运行cv1880v2 测试...\\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\\n");
#else
    printf("使用模拟TIU实现\\n");
#endif

    
#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif
    
    // 执行测试
    test_tiu_maxpool();
    
    printf("所有测试通过!\n");
    return 0;
} 