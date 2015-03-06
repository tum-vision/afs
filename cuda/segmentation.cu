#include "segmentation.cuh"
#include <stdio.h>              // for getting Date and time
#include <time.h>               // contains clock()
#include <iostream>
#include "cutil5.cuh"

using namespace std;

#define BLOCKDIMX 16
#define BLOCKDIMY 16


// update dual variables
__global__ void kernel_grad_ascent(float *u_bar, float *xi, float *psi, float *sum_u, float *g, int width, int height, int n, float lambda, int pitch)
{
    // to find out where thread is located
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // update xi and psi:
    const float tau_xi   = 1.0f / 2;
    const float tau_psi  = 1.0f / n;
    const float proj_xi = g[x + y * pitch] * lambda;

    // iterate over all regions i
    for(int i = 0; i < n; i++)
    {
        // set positions within arrays
        const int pos_u  = x + y * pitch + i * pitch * height;
        const int pos_xi = x + y * pitch + i * pitch * height * 2;

        if(x < width && y < height)
        {
            // update xi: xi = xi - tau_xi * grad(u_bar)
            float xi_pos_xi   = xi[pos_xi];
            float xi_pos_xi_2 = xi[pos_xi + pitch * height];

            if (x < width - 1){
                xi_pos_xi   += tau_xi * (u_bar[pos_u + 1]     - u_bar[pos_u]);
            }
            if (y < height - 1){
                xi_pos_xi_2 += tau_xi * (u_bar[pos_u + pitch] - u_bar[pos_u]);
            }

            // project xi if needed
            float norm_xi = sqrtf( xi_pos_xi   * xi_pos_xi
                                 + xi_pos_xi_2 * xi_pos_xi_2 );
            if(norm_xi > proj_xi){
                xi_pos_xi   = xi_pos_xi   / norm_xi * proj_xi;
                xi_pos_xi_2 = xi_pos_xi_2 / norm_xi * proj_xi;
            }

            // save new values of xi
            xi[pos_xi]                  = xi_pos_xi;
            xi[pos_xi + pitch * height] = xi_pos_xi_2;
            // ================================================
        } // if(x < width && y < height)
    }

    // update psi: psi = psi + tau_psi * (sum_u - 1)
    psi[x + y * pitch] += tau_psi * (sum_u[x + y * pitch] - 1);
}

// update primal variables
__global__ void kernel_grad_descent(float *dataterm, float *u, float *u_bar, float *xi, float *psi, float *sum_u, int width, int height, int n, int pitch)
{
    // to find out where thread is located
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // to check weather thread is within the image
    if(x >= width || y >= height)
        return;

    // update u
    const float tau_u    = 1.0f / 6.0f;
    const float psi_temp = psi[x + y * pitch];

    // initialize sum_u
    float sum_u_temp = 0;

    // iterate over all regions
    for(int i = 0; i < n; i++)
    {
        // set positions within arrays     
        const int pos_u  = x + y * pitch + i * pitch * height;
        const int pos_xi = x + y * pitch + i * pitch * height * 2;


        // determine div_xi
        float div_xi;
        if(x == 0 && y == 0)
            div_xi = 0;
        else if(x == 0)
            div_xi = xi[pos_xi + pitch * height] - xi[pos_xi + pitch * height - pitch];
        else if(y == 0)
            div_xi = xi[pos_xi] - xi[pos_xi - 1];
        else {
            div_xi = xi[pos_xi] - xi[pos_xi - 1]
                   + xi[pos_xi + pitch * height] - xi[pos_xi + pitch * height - pitch];
        }

        // update u: u = u - tau_u * (dataterm + div_xi + 1/n * psi)   -> project u  -> extrapolate u
        float u_temp = u[pos_u];
        float u_old  = u_temp;

        u_temp -= tau_u * (dataterm[pos_u] - div_xi + 1.0f/n * psi_temp);

        if(u_temp > 1)
           u_temp = 1;
        if(u_temp < 0)
           u_temp = 0;

        u[pos_u] = u_temp;

        // determine overrelaxed u and sum_u
        float u_bar_temp = 2.0f * u_temp - u_old;
        u_bar[pos_u] = u_bar_temp;
        sum_u_temp  += u_bar_temp;
    }
    // save sum_u
    sum_u[x + y * pitch] = sum_u_temp;
}

void call_segmentation(float *dataterm, float *g, float *u, int width, int height, int n, float lambda, int &maxSteps, double &time_segm)
{
    // has to be determined to run the algo on the GPU (to know the dimensions on the GPU & to efficiently use it)
    dim3 dimBlock(BLOCKDIMX, BLOCKDIMY);
    dim3 dimGrid;
    size_t pitch;
    // Set grid size (in number of blocks)
    dimGrid.x = (width  % dimBlock.x) ? (width /dimBlock.x + 1) : (width /dimBlock.x);
    dimGrid.y = (height % dimBlock.y) ? (height/dimBlock.y + 1) : (height/dimBlock.y);


    // allocate the memory on the GPU
    float *gpu_dataterm, *gpu_g, *gpu_u, *gpu_u_bar, *gpu_u_prev, *gpu_xi, *gpu_psi;

    cutilSafeCall(cudaMallocPitch((void**)&gpu_dataterm, &pitch, width * sizeof(float), height * n));    // gpu_dataterm
    cutilSafeCall(cudaMemcpy2D(gpu_dataterm, pitch, dataterm, width * sizeof(float), width * sizeof(float), height * n, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_g, &pitch, width * sizeof(float), height));                      // gpu_g
    cutilSafeCall(cudaMemcpy2D(gpu_g, pitch, g, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_u, &pitch, width * sizeof(float), height * n));           // gpu_u
    cutilSafeCall(cudaMemset2D(gpu_u, pitch, 0, width * sizeof(float), height * n));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_u_bar, &pitch, width * sizeof(float), height * n));       // gpu_u_bar
    cutilSafeCall(cudaMemset2D(gpu_u_bar, pitch, 0, width * sizeof(float), height * n));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_u_prev, &pitch, width * sizeof(float), height * n));     // gpu_u_prev
    cutilSafeCall(cudaMemset2D(gpu_u_prev, pitch, 0, width * sizeof(float), height * n));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_xi, &pitch, width * sizeof(float), height * n * 2));   // xi
    cutilSafeCall(cudaMemset2D(gpu_xi, pitch, 0, width * sizeof(float), height * n * 2));

    cutilSafeCall(cudaMallocPitch((void**)&gpu_psi, &pitch, width * sizeof(float), height));                 // psi
    cutilSafeCall(cudaMemset2D(gpu_psi, pitch, 0, width * sizeof(float), height));


    // allocate the memory for sum_u and initialize sum_u with 1
    float* sum_u = new float [width * height];
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            sum_u[x + y * width] = 1;
        }
    }
    float *gpu_sum_u;
    cutilSafeCall(cudaMallocPitch((void**)&gpu_sum_u, &pitch, width * sizeof(float), height));                                              // sum_u
    cutilSafeCall(cudaMemcpy2D(gpu_sum_u, pitch, sum_u, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));


    // run the segmentation (primal-dual-algo) & count the time
    double tstart;
    time_segm = 0.0;    // time measurment variables
    tstart = clock();
    for(int step = 0; step < maxSteps; step++)
    {
        // update xi, psi
        kernel_grad_ascent<<< dimGrid, dimBlock >>>(gpu_u_bar, gpu_xi, gpu_psi, gpu_sum_u, gpu_g, width, height, n, lambda, pitch/sizeof(float));
        cutilSafeCall( cudaThreadSynchronize() );

        // update u
        kernel_grad_descent<<< dimGrid, dimBlock >>>(gpu_dataterm, gpu_u, gpu_u_bar, gpu_xi, gpu_psi, gpu_sum_u, width, height, n, pitch/sizeof(float));
        cutilSafeCall( cudaThreadSynchronize() );
    }
    time_segm += clock() - tstart;         // end
    time_segm = time_segm/CLOCKS_PER_SEC;  // rescale to seconds


    // copy result back to CPU
    cutilSafeCall(cudaMemcpy2D((void*)u, width * sizeof(float), gpu_u, pitch, width * sizeof(float), height * n, cudaMemcpyDeviceToHost));


    // delete all data
    delete sum_u;
    cutilSafeCall(cudaFree(gpu_dataterm));
    cutilSafeCall(cudaFree(gpu_g));
    cutilSafeCall(cudaFree(gpu_u));
    cutilSafeCall(cudaFree(gpu_u_bar));
    cutilSafeCall(cudaFree(gpu_xi));
    cutilSafeCall(cudaFree(gpu_psi));    
    cutilSafeCall(cudaFree(gpu_sum_u));
}
