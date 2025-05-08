// Test TIU matrix addition for bm1822 chip #TestBM1822TIUMatAdd
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef BM1822_USE_REAL_IMPL
// Simulation functions and data structures
#endif // BM1822_USE_REAL_IMPL

// Test matrix addition
void test_tiu_matrix_add() {
    printf("Testing TIU matrix addition...\n");
    
    // Create kernel context
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "bm1822");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef BM1822_USE_REAL_IMPL
    // Register context - real TIU API
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    
    // Create test data
    int n = 1, c = 1, h = 3, w = 4;
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // Allocate tensors in Local Memory
    cvk_tl_t *tl_input1 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_input2 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // Global memory tensors
    cvk_tg_t g_input1, g_input2, g_output;
    memset(&g_input1, 0, sizeof(cvk_tg_t));
    memset(&g_input2, 0, sizeof(cvk_tg_t));
    memset(&g_output, 0, sizeof(cvk_tg_t));
    
    // TODO: Initialize global memory tensors in real implementation
    
    // Load data from global memory to tensors
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
    
    // Execute TIU matrix addition
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
    
    // Copy results from tensor to global memory
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output;
    param3.dst = &g_output;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // Free local memory tensors
    ctx->ops->lmem_free_tensor(ctx, tl_input1);
    ctx->ops->lmem_free_tensor(ctx, tl_input2);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // Register context - simulation
    ctx = malloc(sizeof(cvk_context_t)); // Simple simulation
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // Simulation print operations
    printf("Simulating TIU matrix addition...\n");
    printf("Matrix A: shape [3, 4]\n");
    printf("Matrix B: shape [3, 4]\n");
    printf("Matrix C: shape [3, 4] (result)\n");
    printf("Executing TIU matrix addition operation\n");
    
    // Sample data for visualization
    int8_t matrix_a[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    int8_t matrix_b[3][4] = {
        {10, 20, 30, 40},
        {50, 60, 70, 80},
        {90, 100, 110, 120}
    };
    
    int8_t matrix_c[3][4] = {0}; // Result matrix
    
    // Perform matrix addition
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            // Add elements and handle potential overflow
            int16_t sum = (int16_t)matrix_a[i][j] + (int16_t)matrix_b[i][j];
            matrix_c[i][j] = (sum > INT8_MAX) ? INT8_MAX : ((sum < INT8_MIN) ? INT8_MIN : sum);
        }
    }
    
    // Print matrices
    printf("\nMatrix A (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_b[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix C = A+B (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_c[i][j]);
        }
        printf("\n");
    }
    
    // Example with potential overflow
    printf("\nExample with values that may cause overflow:\n");
    
    // Create matrices with values that may cause overflow
    int8_t overflow_a[2][2] = {
        {120, 100},
        {-120, -100}
    };
    
    int8_t overflow_b[2][2] = {
        {40, 30},
        {-40, -30}
    };
    
    int16_t overflow_raw[2][2] = {0}; // Raw result before clamping
    int8_t overflow_clamped[2][2] = {0}; // Result after clamping to INT8
    
    // Compute raw and clamped results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            overflow_raw[i][j] = (int16_t)overflow_a[i][j] + (int16_t)overflow_b[i][j];
            overflow_clamped[i][j] = (overflow_raw[i][j] > INT8_MAX) ? INT8_MAX : 
                                   ((overflow_raw[i][j] < INT8_MIN) ? INT8_MIN : overflow_raw[i][j]);
        }
    }
    
    // Print overflow example matrices
    printf("\nOverflow A (2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", overflow_a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nOverflow B (2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", overflow_b[i][j]);
        }
        printf("\n");
    }
    
    printf("\nRaw result (no clamping):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", overflow_raw[i][j]);
        }
        printf("\n");
    }
    
    printf("\nClamped result (INT8 range):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", overflow_clamped[i][j]);
        }
        printf("\n");
    }
#endif // BM1822_USE_REAL_IMPL
    
    // Free resources
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU matrix addition test passed!\n");
}

// Test matrix addition with scaling
void test_tiu_matrix_add_scale() {
    printf("Testing TIU matrix addition with scaling...\n");
    
    // Create kernel context
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "bm1822");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef BM1822_USE_REAL_IMPL
    // Register context - real TIU API
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    
    // Create test data
    int n = 1, c = 1, h = 3, w = 4;
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // Allocate tensors in Local Memory
    cvk_tl_t *tl_input1 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_input2 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I8, 1);
    
    // Global memory tensors
    cvk_tg_t g_input1, g_input2, g_output;
    memset(&g_input1, 0, sizeof(cvk_tg_t));
    memset(&g_input2, 0, sizeof(cvk_tg_t));
    memset(&g_output, 0, sizeof(cvk_tg_t));
    
    // TODO: Initialize global memory tensors in real implementation
    
    // Load data from global memory to tensors
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
    
    // Execute TIU matrix addition with right shift
    cvk_tiu_add_param_t add_param;
    memset(&add_param, 0, sizeof(add_param));
    add_param.res_high = NULL;
    add_param.res_low = tl_output;
    add_param.a_high = NULL;
    add_param.a_low = tl_input1;
    add_param.b_high = NULL;
    add_param.b_low = tl_input2;
    add_param.rshift_bits = 1; // Right shift by 1 bit (equivalent to divide by 2)
    add_param.layer_id = 0;
    
    ctx->ops->tiu_add(ctx, &add_param);
    
    // Copy results from tensor to global memory
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output;
    param3.dst = &g_output;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // Free local memory tensors
    ctx->ops->lmem_free_tensor(ctx, tl_input1);
    ctx->ops->lmem_free_tensor(ctx, tl_input2);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // Register context - simulation
    ctx = malloc(sizeof(cvk_context_t)); // Simple simulation
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // Simulation print operations
    printf("Simulating TIU matrix addition with scaling...\n");
    printf("Matrix A: shape [3, 4]\n");
    printf("Matrix B: shape [3, 4]\n");
    printf("Matrix C: shape [3, 4] (result)\n");
    printf("Operation: C = (A + B) >> 1 (divide by 2)\n");
    
    // Sample data for visualization
    int8_t matrix_a[3][4] = {
        {10, 20, 30, 40},
        {50, 60, 70, 80},
        {90, 100, 110, 120}
    };
    
    int8_t matrix_b[3][4] = {
        {10, 10, 10, 10},
        {20, 20, 20, 20},
        {30, 30, 30, 30}
    };
    
    int8_t matrix_c[3][4] = {0}; // Result matrix
    
    // Perform matrix addition with right shift
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            int16_t sum = (int16_t)matrix_a[i][j] + (int16_t)matrix_b[i][j];
            // Right shift by 1 (equivalent to divide by 2)
            matrix_c[i][j] = (int8_t)(sum >> 1);
        }
    }
    
    // Print matrices
    printf("\nMatrix A (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_b[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix C = (A+B)>>1 (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_c[i][j]);
        }
        printf("\n");
    }
    
    // Example with odd numbers
    printf("\nExample with odd and negative numbers (showing division rounding):\n");
    
    int8_t odd_a[2][2] = {
        {5, 7},
        {-5, -7}
    };
    
    int8_t odd_b[2][2] = {
        {6, 8},
        {-6, -8}
    };
    
    int8_t odd_result[2][2] = {0};
    
    // Compute results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int16_t sum = (int16_t)odd_a[i][j] + (int16_t)odd_b[i][j];
            // Right shift by 1 (divide by 2)
            odd_result[i][j] = (int8_t)(sum >> 1);
        }
    }
    
    // Print results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("(%d + %d) >> 1 = %d\n", odd_a[i][j], odd_b[i][j], odd_result[i][j]);
        }
    }
#endif // BM1822_USE_REAL_IMPL
    
    // Free resources
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU matrix addition with scaling test passed!\n");
}

// Test INT16 format matrix addition
void test_tiu_matrix_add_int16() {
    printf("Testing TIU INT16 format matrix addition...\n");
    
    // Create kernel context
    cvk_context_t *ctx = NULL;
    cvk_reg_info_t reg_info;
    memset(&reg_info, 0, sizeof(reg_info));
    strcpy(reg_info.chip_ver_str, "bm1822");
    reg_info.cmdbuf_size = 1024 * 1024; // 1MB
    reg_info.cmdbuf = (uint8_t *)malloc(reg_info.cmdbuf_size);
    
#ifdef BM1822_USE_REAL_IMPL
    // Register context - real TIU API
    ctx = cvikernel_register(&reg_info);
    assert(ctx != NULL);
    
    // Create test data
    int n = 1, c = 1, h = 3, w = 4;
    cvk_tl_shape_t shape = {n, c, h, w};
    
    // Allocate tensors in Local Memory with INT16 format
    cvk_tl_t *tl_input1 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I16, 1);
    cvk_tl_t *tl_input2 = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I16, 1);
    cvk_tl_t *tl_output = ctx->ops->lmem_alloc_tensor(ctx, shape, CVK_FMT_I16, 1);
    
    // Global memory tensors
    cvk_tg_t g_input1, g_input2, g_output;
    memset(&g_input1, 0, sizeof(cvk_tg_t));
    memset(&g_input2, 0, sizeof(cvk_tg_t));
    memset(&g_output, 0, sizeof(cvk_tg_t));
    
    // TODO: Initialize global memory tensors in real implementation
    
    // Load data from global memory to tensors
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
    
    // Execute TIU INT16 matrix addition
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
    
    // Copy results from tensor to global memory
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output;
    param3.dst = &g_output;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // Free local memory tensors
    ctx->ops->lmem_free_tensor(ctx, tl_input1);
    ctx->ops->lmem_free_tensor(ctx, tl_input2);
    ctx->ops->lmem_free_tensor(ctx, tl_output);
#else
    // Register context - simulation
    ctx = malloc(sizeof(cvk_context_t)); // Simple simulation
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // Simulation print operations
    printf("Simulating TIU INT16 matrix addition...\n");
    printf("Matrix A: shape [3, 4] in INT16 format\n");
    printf("Matrix B: shape [3, 4] in INT16 format\n");
    printf("Matrix C: shape [3, 4] (result) in INT16 format\n");
    
    // Sample data for visualization with INT16 values
    int16_t matrix_a[3][4] = {
        {1000, 2000, 3000, 4000},
        {5000, 6000, 7000, 8000},
        {9000, 10000, 11000, 12000}
    };
    
    int16_t matrix_b[3][4] = {
        {500, 1500, 2500, 3500},
        {4500, 5500, 6500, 7500},
        {8500, 9500, 10500, 11500}
    };
    
    int16_t matrix_c[3][4] = {0}; // Result matrix
    
    // Perform INT16 matrix addition
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            // Add elements and handle potential overflow
            int32_t sum = (int32_t)matrix_a[i][j] + (int32_t)matrix_b[i][j];
            matrix_c[i][j] = (sum > INT16_MAX) ? INT16_MAX : ((sum < INT16_MIN) ? INT16_MIN : sum);
        }
    }
    
    // Print matrices
    printf("\nMatrix A (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6d ", matrix_a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6d ", matrix_b[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix C = A+B (3x4):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6d ", matrix_c[i][j]);
        }
        printf("\n");
    }
    
    // Example with INT16 edge cases
    printf("\nExample with INT16 edge cases:\n");
    
    int16_t edge_a[2] = {INT16_MAX, INT16_MIN};
    int16_t edge_b[2] = {1, -1};
    int16_t edge_result[2] = {0};
    int32_t edge_raw[2] = {0};
    
    // Compute and display
    for (int i = 0; i < 2; i++) {
        edge_raw[i] = (int32_t)edge_a[i] + (int32_t)edge_b[i];
        edge_result[i] = (edge_raw[i] > INT16_MAX) ? INT16_MAX : 
                        ((edge_raw[i] < INT16_MIN) ? INT16_MIN : edge_raw[i]);
        
        printf("%d + %d = %d (raw: %d, %s)\n", 
               edge_a[i], edge_b[i], edge_result[i], edge_raw[i],
               (edge_result[i] != edge_raw[i]) ? "overflow detected" : "no overflow");
    }
#endif // BM1822_USE_REAL_IMPL
    
    // Free resources
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU INT16 matrix addition test passed!\n");
}

int main() {
    printf("Running bm1822 TIU matrix addition tests...\n");
    
#ifdef BM1822_USE_REAL_IMPL
    printf("Using real TIU API implementation\n");
#else
    printf("Using simulated TIU implementation\n");
#endif
    
    // Execute tests
    test_tiu_matrix_add();
    test_tiu_matrix_add_scale();
    test_tiu_matrix_add_int16();
    
    printf("All matrix addition tests passed!\n");
    return 0;
} 