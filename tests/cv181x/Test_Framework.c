/* tests/test_template.c

*********Tensor computation testing framework*********
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../src/cv181x/cvkcv181x.h"
#include "../include/cvikernel/cvikernel.h"
#include "../include/cvikernel/cv181x/cv181x_tpu_cfg.h"  // Include hardware configuration macro definitions

// Main test function 
int main() {

    int status = 0;

    // 1. Initialize context
    cvk_context_t *ctx = cvk_cv181x_ops.reset();
    CHECK(ctx != NULL, "Unable to initialize context.");

    // 2. Get command buffer
    void *cmdbuf = cvk_cv181x_ops.acquire_cmdbuf(ctx);
    CHECK(cmdbuf != NULL, "Unable to obtain command buffer.");

    // 3. Tensor allocation
    uint32_t tensor_size = 0x1000; // 4KB
    cvk_tl_t *a_low = cvk_cv181x_ops.lmem_alloc_tensor(ctx, tensor_size);
    cvk_tl_t *b_low = cvk_cv181x_ops.lmem_alloc_tensor(ctx, tensor_size);
    cvk_tl_t *res_low = cvk_cv181x_ops.lmem_alloc_tensor(ctx, tensor_size);
    CHECK(a_low != NULL && b_low != NULL && res_low != NULL, "Unable to allocate tensor。");

    // 4. Initialize tensor
    cvk_tl_shape_t shape = {1, 1, 16, 16}; // Example shape:Batch=1, Channel=1, Height=16, Width=16
    cvk_tl_stride_t stride = {16, 16, 256, 256}; // Example stride

    cvk_cv181x_ops.lmem_init_tensor(ctx, a_low, CVK_FMT_BF16, shape, NULL);
    cvk_cv181x_ops.lmem_init_tensor(ctx, b_low, CVK_FMT_BF16, shape, NULL);
    cvk_cv181x_ops.lmem_init_tensor(ctx, res_low, CVK_FMT_BF16, shape, NULL);

    // 5. Set tensor address (based on hardware configuration)
    a_low->start_address = CV181X_HW_LMEM_START_ADDR + 0x0000;
    b_low->start_address = CV181X_HW_LMEM_START_ADDR + 0x1000;
    res_low->start_address = CV181X_HW_LMEM_START_ADDR + 0x2000; 
    // Set tensor format
    a_low->fmt = CVK_FMT_BF16;
    b_low->fmt = CVK_FMT_BF16;
    res_low->fmt = CVK_FMT_BF16;

    // 6. Allocate host memory and initialize data
    float *host_a = (float *)malloc(sizeof(float) * 16 * 16);
    float *host_b = (float *)malloc(sizeof(float) * 16 * 16);
    float *host_res = (float *)malloc(sizeof(float) * 16 * 16);
    CHECK(host_a != NULL && host_b != NULL && host_res != NULL, "Unable to allocate host memory.");

    // Initialize data according to test requirements (this is an example)
    for (int i = 0; i < 16 * 16; i++) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    // 7. Allocate BF16 host memory
    uint16_t *bf16_a = (uint16_t *)malloc(sizeof(uint16_t) * 16 * 16);
    uint16_t *bf16_b = (uint16_t *)malloc(sizeof(uint16_t) * 16 * 16);
    uint16_t *bf16_res = (uint16_t *)malloc(sizeof(uint16_t) * 16 * 16);
    CHECK(bf16_a != NULL && bf16_b != NULL && bf16_res != NULL, "Unable to allocate BF16 host memory.");

    // 8. convert float to BF16
    for (int i = 0; i < 16 * 16; i++) {
        bf16_a[i] = float_to_bf16(host_a[i]);
        bf16_b[i] = float_to_bf16(host_b[i]);
    }

    // 9. Copy data to device
    CHECK(cvk_copy_tensor_to_device(ctx, a_low, bf16_a) == 0, "Unable to copy data to a_low.");
    CHECK(cvk_copy_tensor_to_device(ctx, b_low, bf16_b) == 0, "Unable to copy data to b_low.");

    // 10. Prepare operation parameters (modify according to specific operations)
    cvk_tiu_test_param_t test_param;
    memset(&test_param, 0, sizeof(test_param));

    // Example: Assume this is an addition operation
    // Adjust the structure and call according to the specific operation (addition, subtraction, etc.)
    test_param.a_low = a_low;
    test_param.a_high = NULL; // BF16 Only use low-rank tensors
    test_param.b_low = b_low;
    test_param.b_high = NULL; // BF16 Only use low-rank tensors
    test_param.res_low = res_low;
    test_param.res_high = NULL; // BF16 Only use low-rank tensors
    test_param.rshift_bits = 0; // Do not shift right
    test_param.layer_id = 1; // Layer ID

    // 11. Call operation function (modify according to specific operation)
    // Example: Call the addition function
    cvk_cv181x_ops.tiu_add(ctx, (cvk_tiu_add_param_t *)&test_param);
    // If it is a subtraction operation, call the corresponding function, such as:
    // cvkcv181x_tiu_sub(ctx, (cvk_tiu_sub_param_t *)&test_param);

    // 12. Synchronous operation (implemented according to the actual API, if needed)
    //For example, wait for the operation to complete
    // ...

    // 13. Copy the result back to the host
    CHECK(cvk_copy_tensor_to_host(ctx, res_low, bf16_res) == 0, "Unable to copy data from res_low to the host.");

    // 14. Convert BF16 backfloat
    for (int i = 0; i < 16 * 16; i++) {
        host_res[i] = bf16_to_float(bf16_res[i]);
    }

    // 15. Verification results (modify according to specific operations)
    int pass = 1;
    for (int i = 0; i < 16 * 16; i++) {
        // Example: The addition operation should be 1 + 2 = 3
        if (fabs(host_res[i] - 3.0f) > 1e-3) {
            printf("The result is incorrect，res[%d] = %f\n", i, host_res[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("Tests passed, all results are as expected.\n");
    } else {
        printf("Test failed, there are unexpected results.\n");
        status = 1;
    }

cleanup:
    // 16. Clean up resources
    if (a_low) cvk_cv181x_ops.lmem_free_tensor(ctx, a_low);
    if (b_low) cvk_cv181x_ops.lmem_free_tensor(ctx, b_low);
    if (res_low) cvk_cv181x_ops.lmem_free_tensor(ctx, res_low);

    if (host_a) free(host_a);
    if (host_b) free(host_b);
    if (host_res) free(host_res);
    if (bf16_a) free(bf16_a);
    if (bf16_b) free(bf16_b);
    if (bf16_res) free(bf16_res);

    // Clear context
    cvk_cv181x_ops.cleanup(ctx);

    if (status) {
        fprintf(stderr, "Test failed.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
    
    return 0;
}