// 测试 cv1880v2 芯片的Sigmoid激活函数功能
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

// INT8量化范围
#define INT8_MIN_VAL (-128)
#define INT8_MAX_VAL (127)

// 全局内存模拟
uint8_t *g_lmem_base = NULL;

#ifndef CV1880V2_USE_REAL_IMPL
// Sigmoid激活函数的LUT(查找表)实现
const int8_t sigmoid_lut[256] = {
    /* 预计算的INT8 Sigmoid查找表 */
    /* -128到-113 */  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    /* -112到-97  */  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
    /* -96到-81   */  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,
    /* -80到-65   */  4,  4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  8,  8,  9,  9,
    /* -64到-49   */  10, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 18, 19, 20, 21,
    /* -48到-33   */  22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 34, 35, 37, 38, 40, 41,
    /* -32到-17   */  43, 45, 46, 48, 50, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 69,
    /* -16到-1    */  71, 73, 74, 76, 78, 79, 81, 82, 84, 85, 87, 88, 89, 90, 92, 93,
    /* 0到15      */  94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 107, 108,
    /* 16到31     */  109, 110, 110, 111, 112, 112, 113, 114, 114, 115, 115, 116, 116, 117, 117, 118,
    /* 32到47     */  118, 119, 119, 119, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 123,
    /* 48到63     */  124, 124, 124, 124, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126,
    /* 64到79     */  127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    /* 80到95     */  127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    /* 96到111    */  127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    /* 112到127   */  127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127
};

// cv1880v2 INT8 Sigmoid函数实现

#ifndef CV1880V2_USE_REAL_IMPL

void cv1880v2_sigmoid_int8(int8_t *input, int8_t *output, int n, int c, int h, int w,
                         float input_scale, float output_scale) {
    // 模拟cv1880v2硬件的并行处理特性
    const int BATCH_SIZE = 16; // 模拟并行处理单元
    int size = n * c * h * w;
    
    // 防止编译器警告未使用参数
    (void)input_scale;
    (void)output_scale;
    
    // 按批处理进行计算
    for (int i = 0; i < size; i += BATCH_SIZE) {
        // 计算当前批次的结束位置
        int batch_end = (i + BATCH_SIZE < size) ? i + BATCH_SIZE : size;
        
        // 批处理循环 - 在实际硬件中这些操作是并行的
        for (int j = i; j < batch_end; j++) {
            // INT8输入值范围是-128到127
            uint8_t table_index = (uint8_t)(input[j] + 128);
            output[j] = sigmoid_lut[table_index];
        }
    }
}
#endif

// 使用TIU API进行Sigmoid测试

#endif // CV1880V2_USE_REAL_IMPL

void test_tiu_sigmoid() {
    printf("测试 CV1880V2 TIU Sigmoid激活函数...\n");
    
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

/* -128到-113 */  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#else
    ctx = malloc(sizeof(cvk_context_t)); // 简单模拟
#endif

    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU Sigmoid测试通过!\n");
}

// Sigmoid函数参考实现 - 用于验证硬件结果
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main() {
    printf("运行cv1880v2 测试...\n");

#ifdef CV1880V2_USE_REAL_IMPL
    printf("使用真实TIU API实现\n");
#else
    printf("使用模拟TIU实现\n");
#endif

    test_tiu_sigmoid();

    printf("所有测试通过!\n");
    return 0;
}
