#ifndef BINARIZATION_CUH
#define BINARIZATION_CUH

#include <cuda_runtime.h>

/** \addtogroup CUDA_CODE
 *  @{
 */

/**
 * @brief CUDA kernel function to binarize the relaxed solution u
 * @param u             relaxed optimized solution
 * @param u_binary      binary solution
 * @param width         width of image
 * @param height        height of image
 * @param n             number of regions(classes)
 * @param pitch         CUDA memory management
 * @param pitchUChar    CUDA memory management
 */
__global__ void kernel_binarize_u(float *u, unsigned char *u_binary, int width, int height, int n, int pitch, int pitchUChar);

/**
 * @brief call function to binarize the solution on CUDA
 * @param u         relaxed optimized solution
 * @param u_binary  binary solution
 * @param width     width of image
 * @param height    height of image
 * @param n         number of regions(classes)
 */
void call_binarization(float *u, unsigned char *u_binary, int width, int height, int n);

/** @} */ // end of group CUDA_CODE

#endif // BINARIZATION_CUH
