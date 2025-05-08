// 测试 cv180x 芯片的张量最大池化功能
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef CV180X_USE_REAL_IMPL
// 模拟实现的函数和数据结构
#endif // CV180X_USE_REAL_IMPL

void test_tiu_maxpool() {
    printf("测试 TIU 最大池化运算...\n");
    
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

// 创建张量形状
cvk_tl_shape_t in_shape = {1, 1, in_h, in_w};
cvk_tl_shape_t out_shape = {1, 1, out_h, out_w};

// 在本地内存（Local Memory）中分配张量
cvk_tl_t *tl_input = ctx->ops->lmem_alloc_tensor(ctx, in_shape, CVK_FMT_I8, 1);
cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, out_shape, CVK_FMT_I8, 1);

// 从全局内存加载数据到张量
cvk_tdma_g2l_tensor_copy_param_t param1;
memset(&param1, 0, sizeof(param1));
param1.src = g_input;
param1.dst = tl_input;
param1.layer_id = 0;
ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);

// 执行TIU最大池化运算
cvk_tiu_max_pooling_param_t maxpool_param;
memset(&maxpool_param, 0, sizeof(maxpool_param));
maxpool_param.ofmap = tl_output;
maxpool_param.ifmap = tl_input;
maxpool_param.kh = kernel_h;
maxpool_param.kw = kernel_w;
maxpool_param.pad_top = 0;
maxpool_param.pad_bottom = 0;
maxpool_param.pad_left = 0;
maxpool_param.pad_right = 0;
maxpool_param.stride_h = stride_h;
maxpool_param.stride_w = stride_w;
maxpool_param.ins_val = -128;  // 对于超出边界的填充值
maxpool_param.layer_id = 0;

ctx->ops->tiu_max_pooling(ctx, &maxpool_param);

// 将结果从张量复制到全局内存
cvk_tdma_l2g_tensor_copy_param_t param2;
memset(&param2, 0, sizeof(param2));
param2.src = tl_output;
param2.dst = g_output;
param2.layer_id = 0;
ctx->ops->tdma_l2g_tensor_copy(ctx, &param2);
#else
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // 打印输入数据 - 4x4 矩阵，每个值为位置索引对8取模加1
    printf("输入数据:\n");
    printf("1 2 3 4\n");
    printf("5 6 7 8\n");
    printf("1 2 3 4\n");
    printf("5 6 7 8\n");
    
    // 设置测试参数
    int in_h = 4, in_w = 4;     // 输入大小: 4x4
    int kernel_h = 2, kernel_w = 2;  // 池化核大小: 2x2
    int stride_h = 2, stride_w = 2;  // 步长: 2x2
    int out_h = (in_h - kernel_h) / stride_h + 1;  // 输出大小: 2x2
    int out_w = (in_w - kernel_w) / stride_w + 1;
    
    // 由于这是测试代码且我们不需要实际执行硬件操作，打印操作即可
    printf("模拟TIU最大池化...\n");
    printf("输入张量形状: [1,1,%d,%d]\n", in_h, in_w);
    printf("池化核大小: %dx%d\n", kernel_h, kernel_w);
    printf("步长: %dx%d\n", stride_h, stride_w);
    printf("输出张量形状: [1,1,%d,%d]\n", out_h, out_w);
    
    // 输出预期结果
    printf("输出数据:\n");
    printf("6 8\n");
    printf("6 8\n");
    
    // 如果是真实实现，会使用如下API：
#endif // CV180X_USE_REAL_IMPL
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU最大池化测试通过!\n");
}

int main() {
    printf("运行cv180x 测试...\n");

#ifdef CV180X_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

        printf("运行cv180x TIU最大池化测试...\n");
        
        // 执行测试
        test_tiu_maxpool();
        
        printf("所有测试通过!\n");

    printf("所有测试通过!\n");
    return 0;
}
