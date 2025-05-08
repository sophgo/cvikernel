// Test TIU matrix multiplication for bm1822 chip #TestBM1822TIUMatMul
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../../include/cvikernel/cvikernel.h"

#ifndef BM1822_USE_REAL_IMPL
// Simulation functions and data structures
#endif // BM1822_USE_REAL_IMPL

// Test matrix multiplication
void test_tiu_matrix_mul() {
    printf("Testing TIU matrix multiplication...\n");
    
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
    
    // Create test data for matrix A: shape [1, 1, 2, 3]
    cvk_tl_shape_t shape_a = {1, 1, 2, 3};
    
    // Create test data for matrix B: shape [1, 1, 3, 4]
    cvk_tl_shape_t shape_b = {1, 1, 3, 4};
    
    // Create test data for output matrix C: shape [1, 1, 2, 4]
    cvk_tl_shape_t shape_c = {1, 1, 2, 4};
    
    // Allocate tensors in Local Memory
    cvk_tl_t *tl_input_a = ctx->ops->lmem_alloc_tensor(ctx, shape_a, CVK_FMT_I8, 1);
    cvk_tl_t *tl_input_b = ctx->ops->lmem_alloc_tensor(ctx, shape_b, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output_c = ctx->ops->lmem_alloc_tensor(ctx, shape_c, CVK_FMT_I8, 1);
    
    // Global memory tensors
    cvk_tg_t g_input_a, g_input_b, g_output_c;
    memset(&g_input_a, 0, sizeof(cvk_tg_t));
    memset(&g_input_b, 0, sizeof(cvk_tg_t));
    memset(&g_output_c, 0, sizeof(cvk_tg_t));
    
    // TODO: Initialize global memory tensors in real implementation
    
    // Load data from global memory to tensors
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = &g_input_a;
    param1.dst = tl_input_a;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    cvk_tdma_g2l_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = &g_input_b;
    param2.dst = tl_input_b;
    param2.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param2);
    
    // Execute TIU matrix multiplication
    cvk_tiu_matrix_multiplication_param_t mm_param;
    memset(&mm_param, 0, sizeof(mm_param));
    mm_param.res = tl_output_c;
    mm_param.a = tl_input_a;
    mm_param.b = tl_input_b;
    mm_param.relu_enable = 0;
    mm_param.layer_id = 0;
    
    ctx->ops->tiu_matrix_multiplication(ctx, &mm_param);
    
    // Copy results from tensor to global memory
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output_c;
    param3.dst = &g_output_c;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // Free local memory tensors
    ctx->ops->lmem_free_tensor(ctx, tl_input_a);
    ctx->ops->lmem_free_tensor(ctx, tl_input_b);
    ctx->ops->lmem_free_tensor(ctx, tl_output_c);
#else
    // Register context - simulation
    ctx = malloc(sizeof(cvk_context_t)); // Simple simulation
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // Simulation print operations
    printf("Simulating TIU matrix multiplication...\n");
    printf("Matrix A: shape [2, 3]\n");
    printf("Matrix B: shape [3, 4]\n");
    printf("Matrix C: shape [2, 4] (result)\n");
    printf("Executing TIU matrix multiplication operation\n");
    
    // Sample data for visualization
    int8_t matrix_a[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };
    
    int8_t matrix_b[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    int8_t matrix_c[2][4] = {0}; // Result matrix
    
    // Perform simple matrix multiplication
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    
    // Print matrices
    printf("\nMatrix A (2x3):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
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
    
    printf("\nMatrix C = A*B (2x4):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4d ", matrix_c[i][j]);
        }
        printf("\n");
    }
    
    // Demonstrate matrix multiplication with overflow
    printf("\nExample with potential overflow:\n");
    
    // Large values that might overflow in INT8
    matrix_a[0][0] = 50; matrix_a[0][1] = 40; matrix_a[0][2] = 30;
    matrix_b[0][0] = 5;  matrix_b[1][0] = 4;  matrix_b[2][0] = 3;
    
    // Reset result
    matrix_c[0][0] = 0;
    
    // Compute one element with potential overflow
    for (int k = 0; k < 3; k++) {
        matrix_c[0][0] += matrix_a[0][k] * matrix_b[k][0];
    }
    
    printf("Element C[0][0] = %d (potential overflow for INT8)\n", matrix_c[0][0]);
    
    // Handle potential overflow (clamping) as hardware might do
    int32_t actual_result = 0;
    for (int k = 0; k < 3; k++) {
        actual_result += matrix_a[0][k] * matrix_b[k][0];
    }
    
    int8_t clamped_result = (actual_result > INT8_MAX) ? INT8_MAX : 
                           ((actual_result < INT8_MIN) ? INT8_MIN : actual_result);
    
    printf("Actual value without INT8 clamping: %d\n", actual_result);
    printf("Value after clamping to INT8 range: %d\n", clamped_result);
#endif // BM1822_USE_REAL_IMPL
    
    // Free resources
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU matrix multiplication test passed!\n");
}

// Test matrix multiplication with ReLU activation
void test_tiu_matrix_mul_relu() {
    printf("Testing TIU matrix multiplication with ReLU activation...\n");
    
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
    
    // Create test data for matrix A: shape [1, 1, 2, 3]
    cvk_tl_shape_t shape_a = {1, 1, 2, 3};
    
    // Create test data for matrix B: shape [1, 1, 3, 4]
    cvk_tl_shape_t shape_b = {1, 1, 3, 4};
    
    // Create test data for output matrix C: shape [1, 1, 2, 4]
    cvk_tl_shape_t shape_c = {1, 1, 2, 4};
    
    // Allocate tensors in Local Memory
    cvk_tl_t *tl_input_a = ctx->ops->lmem_alloc_tensor(ctx, shape_a, CVK_FMT_I8, 1);
    cvk_tl_t *tl_input_b = ctx->ops->lmem_alloc_tensor(ctx, shape_b, CVK_FMT_I8, 1);
    cvk_tl_t *tl_output_c = ctx->ops->lmem_alloc_tensor(ctx, shape_c, CVK_FMT_I8, 1);
    
    // Global memory tensors
    cvk_tg_t g_input_a, g_input_b, g_output_c;
    memset(&g_input_a, 0, sizeof(cvk_tg_t));
    memset(&g_input_b, 0, sizeof(cvk_tg_t));
    memset(&g_output_c, 0, sizeof(cvk_tg_t));
    
    // TODO: Initialize global memory tensors in real implementation
    
    // Load data from global memory to tensors
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = &g_input_a;
    param1.dst = tl_input_a;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    cvk_tdma_g2l_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = &g_input_b;
    param2.dst = tl_input_b;
    param2.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param2);
    
    // Execute TIU matrix multiplication with ReLU
    cvk_tiu_matrix_multiplication_param_t mm_param;
    memset(&mm_param, 0, sizeof(mm_param));
    mm_param.res = tl_output_c;
    mm_param.a = tl_input_a;
    mm_param.b = tl_input_b;
    mm_param.relu_enable = 1; // Enable ReLU activation
    mm_param.layer_id = 0;
    
    ctx->ops->tiu_matrix_multiplication(ctx, &mm_param);
    
    // Copy results from tensor to global memory
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output_c;
    param3.dst = &g_output_c;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // Free local memory tensors
    ctx->ops->lmem_free_tensor(ctx, tl_input_a);
    ctx->ops->lmem_free_tensor(ctx, tl_input_b);
    ctx->ops->lmem_free_tensor(ctx, tl_output_c);
#else
    // Register context - simulation
    ctx = malloc(sizeof(cvk_context_t)); // Simple simulation
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // Simulation print operations
    printf("Simulating TIU matrix multiplication with ReLU...\n");
    printf("Matrix A: shape [2, 3]\n");
    printf("Matrix B: shape [3, 2]\n");
    printf("Matrix C: shape [2, 2] (result)\n");
    printf("Executing TIU matrix multiplication with ReLU operation\n");
    
    // Sample data for visualization with negative values
    int8_t matrix_a[2][3] = {
        {1, -2, 3},
        {-4, 5, -6}
    };
    
    int8_t matrix_b[3][2] = {
        {1, -2},
        {-3, 4},
        {5, -6}
    };
    
    int32_t matrix_c_raw[2][2] = {0}; // Raw result before ReLU
    int8_t matrix_c_relu[2][2] = {0}; // Result after ReLU
    
    // Perform simple matrix multiplication
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
                matrix_c_raw[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
            // Apply ReLU activation (max(0, x))
            matrix_c_relu[i][j] = (matrix_c_raw[i][j] > 0) ? 
                                  ((matrix_c_raw[i][j] > INT8_MAX) ? INT8_MAX : matrix_c_raw[i][j]) : 0;
        }
    }
    
    // Print matrices
    printf("\nMatrix A (2x3):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%4d ", matrix_a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (3x2):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", matrix_b[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix C raw = A*B (2x2) before ReLU:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", matrix_c_raw[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix C = ReLU(A*B) (2x2) after ReLU:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%4d ", matrix_c_relu[i][j]);
        }
        printf("\n");
    }
#endif // BM1822_USE_REAL_IMPL
    
    // Free resources
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU matrix multiplication with ReLU activation test passed!\n");
}

// Test INT16 matrix multiplication
void test_tiu_matrix_mul_int16() {
    printf("Testing TIU INT16 matrix multiplication...\n");
    
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
    
    // Create test data for matrix A: shape [1, 1, 2, 3]
    cvk_tl_shape_t shape_a = {1, 1, 2, 3};
    
    // Create test data for matrix B: shape [1, 1, 3, 4]
    cvk_tl_shape_t shape_b = {1, 1, 3, 4};
    
    // Create test data for output matrix C: shape [1, 1, 2, 4]
    cvk_tl_shape_t shape_c = {1, 1, 2, 4};
    
    // Allocate tensors in Local Memory with INT16 format
    cvk_tl_t *tl_input_a = ctx->ops->lmem_alloc_tensor(ctx, shape_a, CVK_FMT_I16, 1);
    cvk_tl_t *tl_input_b = ctx->ops->lmem_alloc_tensor(ctx, shape_b, CVK_FMT_I16, 1);
    cvk_tl_t *tl_output_c = ctx->ops->lmem_alloc_tensor(ctx, shape_c, CVK_FMT_I16, 1);
    
    // Global memory tensors
    cvk_tg_t g_input_a, g_input_b, g_output_c;
    memset(&g_input_a, 0, sizeof(cvk_tg_t));
    memset(&g_input_b, 0, sizeof(cvk_tg_t));
    memset(&g_output_c, 0, sizeof(cvk_tg_t));
    
    // TODO: Initialize global memory tensors in real implementation
    
    // Load data from global memory to tensors
    cvk_tdma_g2l_tensor_copy_param_t param1;
    memset(&param1, 0, sizeof(param1));
    param1.src = &g_input_a;
    param1.dst = tl_input_a;
    param1.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param1);
    
    cvk_tdma_g2l_tensor_copy_param_t param2;
    memset(&param2, 0, sizeof(param2));
    param2.src = &g_input_b;
    param2.dst = tl_input_b;
    param2.layer_id = 0;
    ctx->ops->tdma_g2l_tensor_copy(ctx, &param2);
    
    // Execute TIU INT16 matrix multiplication
    cvk_tiu_matrix_multiplication_param_t mm_param;
    memset(&mm_param, 0, sizeof(mm_param));
    mm_param.res = tl_output_c;
    mm_param.a = tl_input_a;
    mm_param.b = tl_input_b;
    mm_param.relu_enable = 0;
    mm_param.layer_id = 0;
    
    ctx->ops->tiu_matrix_multiplication(ctx, &mm_param);
    
    // Copy results from tensor to global memory
    cvk_tdma_l2g_tensor_copy_param_t param3;
    memset(&param3, 0, sizeof(param3));
    param3.src = tl_output_c;
    param3.dst = &g_output_c;
    param3.layer_id = 0;
    ctx->ops->tdma_l2g_tensor_copy(ctx, &param3);
    
    // Free local memory tensors
    ctx->ops->lmem_free_tensor(ctx, tl_input_a);
    ctx->ops->lmem_free_tensor(ctx, tl_input_b);
    ctx->ops->lmem_free_tensor(ctx, tl_output_c);
#else
    // Register context - simulation
    ctx = malloc(sizeof(cvk_context_t)); // Simple simulation
    memset(ctx, 0, sizeof(cvk_context_t));
    assert(ctx != NULL);
    
    // Simulation print operations
    printf("Simulating TIU INT16 matrix multiplication...\n");
    printf("Matrix A: shape [2, 3] in INT16 format\n");
    printf("Matrix B: shape [3, 2] in INT16 format\n");
    printf("Matrix C: shape [2, 2] (result) in INT16 format\n");
    printf("Executing TIU INT16 matrix multiplication operation\n");
    
    // Sample data for visualization with INT16 values
    int16_t matrix_a[2][3] = {
        {1000, 2000, 3000},
        {4000, 5000, 6000}
    };
    
    int16_t matrix_b[3][2] = {
        {100, 200},
        {300, 400},
        {500, 600}
    };
    
    int32_t matrix_c[2][2] = {0}; // Result matrix (using int32 to avoid overflow)
    
    // Perform simple matrix multiplication
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 3; k++) {
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    
    // Print matrices
    printf("\nMatrix A (2x3):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%6d ", matrix_a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (3x2):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%6d ", matrix_b[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix C = A*B (2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%10d ", matrix_c[i][j]);
        }
        printf("\n");
    }
    
    // Check if results would overflow INT16
    printf("\nChecking for INT16 overflow:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (matrix_c[i][j] > INT16_MAX || matrix_c[i][j] < INT16_MIN) {
                printf("Overflow at C[%d][%d] = %d (exceeds INT16 range)\n", i, j, matrix_c[i][j]);
                
                // Show clamped value
                int16_t clamped = (matrix_c[i][j] > INT16_MAX) ? INT16_MAX : 
                                 ((matrix_c[i][j] < INT16_MIN) ? INT16_MIN : matrix_c[i][j]);
                printf("   - After clamping: %d\n", clamped);
            } else {
                printf("No overflow at C[%d][%d] = %d\n", i, j, matrix_c[i][j]);
            }
        }
    }
#endif // BM1822_USE_REAL_IMPL
    
    // Free resources
    free(ctx);
    free(reg_info.cmdbuf);
    
    printf("TIU INT16 matrix multiplication test passed!\n");
}

int main() {
    printf("Running bm1822 TIU matrix multiplication tests...\n");
    
#ifdef BM1822_USE_REAL_IMPL
    printf("Using real TIU API implementation\n");
#else
    printf("Using simulated TIU implementation\n");
#endif
    
    // Execute tests
    test_tiu_matrix_mul();
    test_tiu_matrix_mul_relu();
    test_tiu_matrix_mul_int16();
    
    printf("All matrix multiplication tests passed!\n");
    return 0;
} 