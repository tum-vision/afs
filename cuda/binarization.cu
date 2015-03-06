#include "binarization.cuh"
#include <iostream>
#include <math.h>
#include "cutil5.cuh"

using namespace std;

#define BLOCKDIMX 16
#define BLOCKDIMY 16


// get binarized u
__global__ void kernel_binarize_u(float *u, unsigned char *u_binary, int width, int height, int n, int pitch, int pitchUChar)
{
    // to find out where thread is located
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // to check weather thread is within the image
    if(x >= width || y >= height)
        return;

    // binarize u
    float u_max = -INFINITY;
    unsigned char index = 0;
    for(int i = 0; i < n; i++){
        if(u_max < u[x + y * pitch + i * pitch * height]){
            u_max = u[x + y * pitch + i * pitch * height];
            index = i;
        }
    }
    u_binary[x + y * pitchUChar] = index;
}



// transfer variables usw to GPU
void call_binarization(float *u, unsigned char *u_binary, int width, int height, int n)
{
    // has to be determined to run the algo on the GPU (to know the dimensions on the GPU & to efficiently use it)
    dim3 dimBlock(BLOCKDIMX, BLOCKDIMY);
    dim3 dimGrid;
    size_t pitch;
    // Set grid size (in number of blocks)
    dimGrid.x = (width % dimBlock.x) ? (width/dimBlock.x + 1) : (width/dimBlock.x);
    dimGrid.y = (height % dimBlock.y) ? (height/dimBlock.y + 1) : (height/dimBlock.y);


    // allocate the memory on the GPU
    float *gpu_u;
    unsigned char *gpu_u_binary;
    size_t pitchUChar;

    cutilSafeCall(cudaMallocPitch((void**)&gpu_u, &pitch, width * sizeof(float), height * n));                   // u
    cutilSafeCall(cudaMemcpy2D(gpu_u, pitch, u, width * sizeof(float), width * sizeof(float), height * n, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_u_binary, &pitchUChar, width * sizeof(unsigned char), height));   // u_binary
    cutilSafeCall(cudaMemset2D(gpu_u_binary, pitchUChar, 0, width * sizeof(unsigned char), height));


    // call the kernel function
    kernel_binarize_u<<< dimGrid, dimBlock >>>(gpu_u, gpu_u_binary, width, height, n,
                                                        pitch/sizeof(float), pitchUChar/sizeof(unsigned char));
    cutilSafeCall( cudaThreadSynchronize() );


    // copy result back to CPU
    cutilSafeCall(cudaMemcpy2D((void*)u_binary, width * sizeof(unsigned char), gpu_u_binary, pitchUChar, width * sizeof(unsigned char), height,     cudaMemcpyDeviceToHost));

    // delete all data
    cutilSafeCall(cudaFree(gpu_u));
    cutilSafeCall(cudaFree(gpu_u_binary));
}
