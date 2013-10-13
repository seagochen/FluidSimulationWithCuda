/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <First>       Oct 12, 2013
* <Last>		Oct 12, 2013
* <File>        CUDA_Routine.cpp
*/

#include "Macro_Definitions.h"

#define _CUDA_ROUTINE_CPP_

#if GPU_ON

#include <SGE\SGUtils.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__global__ void add_source_kernel ( float *ptr_out, float *ptr_in )
{
	// Get index of GPU-thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int ind = GPUIndex(i, j);

	// Yield value
	ptr_out[ind] += DELTA_TIME * ptr_in[ind];
};


__global__ void set_bnd_kernel ( float *grid_out, int boundary )
{
#define is ==
	// Get index of GPU-thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// Boundary condition
	if ( i >= 1 && i <= GRIDS_WITHOUT_GHOST && j >= 1 && j <= GRIDS_WITHOUT_GHOST )
	{
		// Slove line (0, y)
		grid_out[GPUIndex(0, j)]  = boundary is 1 ? -grid_out[GPUIndex(1, j)] : grid_out[GPUIndex(1, j)];
		// Slove line (65, y)
		grid_out[GPUIndex(65, j)] = boundary is 1 ? -grid_out[GPUIndex(64,j)] : grid_out[GPUIndex(64,j)];
		// Slove line (x, 0)
		grid_out[GPUIndex(i, 0)]  = boundary is 2 ? -grid_out[GPUIndex(i, 1)] : grid_out[GPUIndex(i, 1)];
		// Slove line (x, 65)
		grid_out[GPUIndex(i, 65)] = boundary is 2 ? -grid_out[GPUIndex(i,64)] : grid_out[GPUIndex(i,64)];
	}
	// Slove ghost cell (0, 0)
	grid_out[GPUIndex(0, 0)] = 0.5f * ( grid_out[GPUIndex(1, 0)]  + grid_out[GPUIndex(0, 1)] );
	// Slove ghost cell (0, 65)
	grid_out[GPUIndex(0, 65)] = 0.5f * ( grid_out[GPUIndex(1, 65)] + grid_out[GPUIndex(0, 64)] );
	// Slove ghost cell (65, 0)
	grid_out[GPUIndex(65, 0)] = 0.5f * ( grid_out[GPUIndex(64, 0)] + grid_out[GPUIndex(65, 1)] );
	// Slove ghost cell (65, 65)
	grid_out[GPUIndex(65, 65)] = 0.5f * ( grid_out[GPUIndex(64, 65)] + grid_out[GPUIndex(65, 64)]);

#undef is
}


__global__ void lin_solve_kernel ( float *grid_inout, float *grid0_in, int boundary, float a, float c )
{
	// Get index of GPU-thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ( i >= 1 && i <= GRIDS_WITHOUT_GHOST && j >= 1 && j <= GRIDS_WITHOUT_GHOST )
	{	
		grid_inout[GPUIndex(i,j)] = (grid0_in[GPUIndex(i,j)] + a * ( grid_inout[GPUIndex(i-1,j)] + 
			grid_inout[GPUIndex(i+1,j)] + grid_inout[GPUIndex(i,j-1)] + grid_inout[GPUIndex(i,j+1)] ) ) / c;	
	}
}


__global__ void advect_kernel(float *density_out, float *density0_in, float *u_in, float *v_in, float dt0)
{
	// Get index of GPU-thread
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1;

	if ( i >= 1 && i <= GRIDS_WITHOUT_GHOST && j >= 1 && j <= GRIDS_WITHOUT_GHOST )
	{
		x = i - dt0 * u_in[GPUIndex(i,j)];
		y = j - dt0 * v_in[GPUIndex(i,j)];
		if (x < 0.5f) x = 0.5f;
		if (x > GRIDS_WITHOUT_GHOST + 0.5f) x = GRIDS_WITHOUT_GHOST+0.5f;

		i0 = (int)x; 
		i1 = i0+1;
		
		if (y < 0.5f) y=0.5f;
		if (y > GRIDS_WITHOUT_GHOST+0.5f) y = GRIDS_WITHOUT_GHOST+0.5f;
		
		j0 = (int)y;
		j1 = j0 + 1;
		s1 = x - i0;
		s0 = 1 - s1;
		t1 = y - j0;
		t0 = 1 - t1;

		density_out[GPUIndex(i,j)] = s0 * ( t0 * density0_in[GPUIndex(i0,j0)] +
			t1 * density0_in[GPUIndex(i0,j1)]) + s1 * ( t0 * density0_in[GPUIndex(i1,j0)] + 
			t1 * density0_in[GPUIndex(i1,j1)]);
	}
};


cudaError_t cuda_add_source ( float *grid, float *grid0 )
{		
	// Define the computing unit size
	dim3 block_size;
	dim3 grid_size;
	block_size.x = 16;
	block_size.y = 16;
	grid_size.x  = ENTIRE_GRIDS_NUMBER / block_size.x;
	grid_size.y  = ENTIRE_GRIDS_NUMBER / block_size.y;

	size_t size = ENTIRE_GRIDS_NUMBER * ENTIRE_GRIDS_NUMBER;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_grid, grid, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_grid0, grid0, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	add_source_kernel<<<grid_size, block_size>>>(dev_grid, dev_grid0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "add_source_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(grid, dev_grid, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:    
    return cudaStatus;
};


cudaError_t cuda_lin_solve (float *grid, float *grid0, int boundary, float a, float c)
{
	// Define the computing unit size
	dim3 block_size;
	dim3 grid_size;
	block_size.x = 16;
	block_size.y = 16;
	grid_size.x  = ENTIRE_GRIDS_NUMBER / block_size.x;
	grid_size.y  = ENTIRE_GRIDS_NUMBER / block_size.y;

	size_t size = ENTIRE_GRIDS_NUMBER * ENTIRE_GRIDS_NUMBER;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_grid, grid, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_grid0, grid0, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	for (int i=0; i<20; i++)
	{
		lin_solve_kernel<<<grid_size, block_size>>>(dev_grid, dev_grid0, boundary, a, c);
	}
	set_bnd_kernel<<<grid_size, block_size>>> (dev_grid, boundary);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "add_source_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(grid, dev_grid, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:    
    return cudaStatus;
}


void cuda_diffuse ( float *grid, float *grid0, int boundary, float diff )
{
	float a=DELTA_TIME*diff*GRIDS_WITHOUT_GHOST*GRIDS_WITHOUT_GHOST;
	cuda_lin_solve ( grid, grid0, boundary, a, 1+4*a );
}


cudaError_t cuda_advect( float *density, float *density0, float *u, float *v,  int boundary )
{
	// Define the computing unit size
	dim3 block_size;
	dim3 grid_size;
	block_size.x = 16;
	block_size.y = 16;
	grid_size.x  = ENTIRE_GRIDS_NUMBER / block_size.x;
	grid_size.y  = ENTIRE_GRIDS_NUMBER / block_size.y;

	size_t size = ENTIRE_GRIDS_NUMBER * ENTIRE_GRIDS_NUMBER;

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_den, density, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_den0, density0, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_u, u, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	float dt0 = DELTA_TIME*GRIDS_WITHOUT_GHOST;
	advect_kernel<<<grid_size, block_size>>>(dev_den, dev_den0, dev_u, dev_v, dt0);
	set_bnd_kernel<<<grid_size, block_size>>>(dev_den, boundary);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "add_source_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(density, dev_den, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	

Error:    
    return cudaStatus;
}

void set_bnd ( int GridSize, int boundary, float * grid )
{
	int i;

	for ( i=1 ; i<=GridSize ; i++ ) {
		grid[Index(0  ,i)] = boundary==1 ? -grid[Index(1,i)] : grid[Index(1,i)];
		grid[Index(GridSize+1,i)] = boundary==1 ? -grid[Index(GridSize,i)] : grid[Index(GridSize,i)];
		grid[Index(i,0  )] = boundary==2 ? -grid[Index(i,1)] : grid[Index(i,1)];
		grid[Index(i,GridSize+1)] = boundary==2 ? -grid[Index(i,GridSize)] : grid[Index(i,GridSize)];
	}
	grid[Index(0  ,0  )] = 0.5f*(grid[Index(1,0  )]+grid[Index(0  ,1)]);
	grid[Index(0  ,GridSize+1)] = 0.5f*(grid[Index(1,GridSize+1)]+grid[Index(0  ,GridSize)]);
	grid[Index(GridSize+1,0  )] = 0.5f*(grid[Index(GridSize,0  )]+grid[Index(GridSize+1,1)]);
	grid[Index(GridSize+1,GridSize+1)] = 0.5f*(grid[Index(GridSize,GridSize+1)]+grid[Index(GridSize+1,GridSize)]);
}

void cuda_project ( float * u, float * v, float * p, float * div )
{
	int i, j;

	for ( i=1 ; i<=GridSize ; i++ )
	{
		for ( j=1 ; j<=GridSize ; j++ )
		{
			div[Index(i,j)] = -0.5f*(u[Index(i+1,j)]-u[Index(i-1,j)]+v[Index(i,j+1)]-v[Index(i,j-1)])/GridSize;		
			p[Index(i,j)] = 0;
		}
	}	
	set_bnd ( GridSize, 0, div ); set_bnd ( GridSize, 0, p );

	cuda_lin_solve ( p, div, 0, 1, 4 );

	for ( i=1 ; i<=GridSize ; i++ )
	{
		for ( j=1 ; j<=GridSize ; j++ ) 
		{
			u[Index(i,j)] -= 0.5f*GridSize*(p[Index(i+1,j)]-p[Index(i-1,j)]);
			v[Index(i,j)] -= 0.5f*GridSize*(p[Index(i,j+1)]-p[Index(i,j-1)]);
		}
	}
	set_bnd ( GridSize, 1, u ); set_bnd ( GridSize, 2, v );
}

void dens_step ( float *grid, float *grid0, float *u, float *v )
{
	cuda_add_source(grid, grid0);
	SWAP ( grid0, grid ); cuda_diffuse ( grid, grid0, 0, DIFFUSION );
	SWAP ( grid0, grid ); cuda_advect ( grid, grid0, u, v, 0 );
}


void vel_step ( float *u, float *v, float *u0, float *v0 )
{
	cuda_add_source ( u, u0 ); cuda_add_source ( v, v0 );
	SWAP ( u0, u ); cuda_diffuse ( u, u0, 1, VISCOSITY );
	SWAP ( v0, v ); cuda_diffuse ( v, v0, 2, VISCOSITY );
	cuda_project ( u, v, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );
	cuda_advect ( u, u0, u0, v0, 1 ); cuda_advect ( v, v0, u0, v0, 2 );
	cuda_project ( u, v, u0, v0 );
}

#endif