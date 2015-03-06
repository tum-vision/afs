#ifndef SEGMENTATION_CUH
#define SEGMENTATION_CUH


#include <cuda_runtime.h>

/** @defgroup CUDA_CODE CUDA_PRIMALDUAL_OPTIMIZATION
 *  This group consists of CUDA kernel and c-call functions for primal-dual algorithm
 *  @{
 */

/**
 * @brief CUDA kernel function for gradient ascent
 * @param u_bar     overrelaxation of u
 * @param xi        dual variable
 * @param psi       Lagrange multiplier for simplex constraint
 * @param sum_u     variable for the simplex constraint
 * @param g         edge detection function
 * @param width     width of the image
 * @param height    height of the image
 * @param n         number of labels(classes)
 * @param lambda    weighting parameter
 * @param pitch     CUDA memory management
 */
__global__ void kernel_grad_ascent(float *u_bar, float *xi, float *psi, float *sum_u, float *g, int width, int height, int n, float lambda, int pitch);

/**
 * @brief CUDA kernel function for gradient descent
 * @param dataterm  dataterm
 * @param u         region indicator variable
 * @param u_bar     overrelaxation of u
 * @param xi        dual variable
 * @param psi       Lagrange multiplier for simplex constraint
 * @param sum_u     variable for the simplex constraint
 * @param width     width of the image
 * @param height    height of the image
 * @param n         number of labels(classes)
 * @param pitch     CUDA memory management
 */
__global__ void kernel_grad_descent(float *dataterm, float *u, float *u_bar, float *xi, float *psi, float *sum_u, int width, int height, int n, int pitch);

/**
 * @brief call function to run segmentation on gpu
 * @param dataterm  dataterm
 * @param g         edge detection function
 * @param u         region indicator variable
 * @param width     width of the image
 * @param height    height of the image
 * @param n         number of labels(classes)
 * @param lambda    weighting parameter
 * @param maxSteps  (maximum) number of iterations
 * @param time_segm measures the runtime
 */
void call_segmentation(float *dataterm, float *g, float *u, int width, int height, int n, float lambda, int &maxSteps, double &time_segm);

/** @} */ // end of group CUDA_CODE

#endif // SEGMENTATION_CUH
