#include <cuda_runtime.h>
#include <stdio.h>

#ifndef cutilSafeCall
#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)

/** \addtogroup CUDA_CODE
 *  @{
 */

/**
 * @brief call function to print out CUDA errors
 * @param err   CUDA error
 * @param file  output file
 * @param line  output line
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line){
  if(cudaSuccess != err) {
    fprintf(stderr,"%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
          file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}

/** @} */ // end of group CUDA_CODE

#endif
