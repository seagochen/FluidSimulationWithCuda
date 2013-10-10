#include "CFD_FuncPrototypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <SGE\SGUtils.h>

#define CUDA_Release(ptr) {if (ptr != 0) cudaFree(ptr);}
#define Error_Free(ptr0, ptr1, ptr2, ptr3) {CUDA_Release(ptr0); CUDA_Release(ptr1); CUDA_Release(ptr2); CUDA_Release(ptr3);}

#define IX(i,j) ((i)+(GridSize+2)*(j))
#define SWAP(grid0,grid) {float * tmp=grid0;grid0=grid;grid=tmp;}

__global__ void add_source_kernel ( float * grid_out, float * src_in, float dt_in )
{
	int i = threadIdx.x;
	grid_out[i] += dt_in*src_in[i];
}


__global__ void set_bnd_kernel ( int GridSize, int boundary, float * grid )
{
	int i = threadIdx.x;

	grid[IX(0  ,i)] = boundary==1 ? -grid[IX(1,i)] : grid[IX(1,i)];
	grid[IX(GridSize+1,i)] = boundary==1 ? -grid[IX(GridSize,i)] : grid[IX(GridSize,i)];
	grid[IX(i,0  )] = boundary==2 ? -grid[IX(i,1)] : grid[IX(i,1)];
	grid[IX(i,GridSize+1)] = boundary==2 ? -grid[IX(i,GridSize)] : grid[IX(i,GridSize)];

	grid[IX(0  ,0  )] = 0.5f*(grid[IX(1,0  )]+grid[IX(0  ,1)]);
	grid[IX(0  ,GridSize+1)] = 0.5f*(grid[IX(1,GridSize+1)]+grid[IX(0  ,GridSize)]);
	grid[IX(GridSize+1,0  )] = 0.5f*(grid[IX(GridSize,0  )]+grid[IX(GridSize+1,1)]);
	grid[IX(GridSize+1,GridSize+1)] = 0.5f*(grid[IX(GridSize,GridSize+1)]+grid[IX(GridSize+1,GridSize)]);
}


__global__ void lin_solve_kernel ( int GridSize, int boundary, float * grid, float * grid0, float a, float c )
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int index = index_y * grid_width + index_x;

	grid[index] = (grid0[index] + a*(grid[IX(i-1,j)]+grid[IX(i+1,j)]+grid[IX(i,j-1)]+grid[IX(i,j+1)]))/c;
}


void diffuse ( int GridSize, int boundary, float * grid, float * grid0, float diff, float dt )
{
	float a=dt*diff*GridSize*GridSize;
	lin_solve ( GridSize, boundary, grid, grid0, a, 1+4*a );
}


cudaError_t cuda_dens_step( int GridSize, float *grid, float *grid0, float *u, float *v, float diff, float dt )
{
	float *grid_cuda = 0;
	float *grid0_cuda= 0;
	float *u_cuda = 0;
	float *v_cuda = 0;

	int size = (GridSize+2)*(GridSize+2);

	cudaError_t cuda_status;

	// create two dimensional 4x4 thread blocks
	dim3 block_size;
	block_size.x = 64 + 2;
	block_size.y = 64 + 2;
	
	// configure a two dimensional grid as well
	dim3 grid_size;
	grid_size.x = 66 / block_size.x;
	grid_size.y = 66 / block_size.y;

	// Choose which GPU to run on, change this on a multi-GPU system
	cuda_status = cudaSetDevice(0);
	if ( cuda_status != cudaError::cudaSuccess ) {
		ErrorMSG("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Allocate GPU buffers for those vectors
	if ( (cuda_status = cudaMalloc((void**)&grid_cuda, size * sizeof(float))) != cudaError::cudaSuccess ) 
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	if ( (cuda_status = cudaMalloc((void**)&grid0_cuda, size * sizeof(float))) != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	if ( (cuda_status = cudaMalloc((void**)&u_cuda, size * sizeof(float))) != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	if ( (cuda_status = cudaMalloc((void**)&v_cuda, size * sizeof(float))) != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Copy input vectors from host memory to GPU buffers
	// Do add_source_kernel first
	cuda_status = cudaMemcpy(grid0_cuda, grid0, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if ( cuda_status != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMemcpy failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Launch a kernel on GPU with one thread for each element
	add_source_kernel<<<1, size>>>(grid_cuda, grid0_cuda, dt);

	// Check for any errors launching the kernel
	cuda_status = cudaGetLastError();
	if ( cuda_status != cudaError::cudaSuccess ) 
	{
		fprintf(stderr, "add_source_kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
	cuda_status = cudaDeviceSynchronize();
	if ( cuda_status != cudaError::cudaSuccess )
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching kernel!\n",
			cuda_status, cudaGetErrorString(cuda_status));
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Copy output vector from GPU buffer to host memory
	cuda_status = cudaMemcpy(grid, grid_cuda, size*sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	if ( cuda_status != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMemcpy failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	SWAP ( grid0, grid );

	return cuda_status;
};

cudaError_t cuda_vel_step( int GridSize, float * u, float * v, float * u0, float * v0, float visc, float dt )
{
	float *u_cuda = 0;
	float *v_cuda = 0;
	float *u0_cuda= 0;
	float *v0_cuda= 0;

	cudaError cuda_status;

	// Choose which GPU to run on, change this on a multi-GPU system
	cuda_status = cudaSetDevice(0);
	if ( cuda_status != cudaError::cudaSuccess ) {
		ErrorMSG("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		CUDA_Release(u_cuda);
		CUDA_Release(v_cuda);
		CUDA_Release(u0_cuda);
		CUDA_Release(v0_cuda);
	}
};