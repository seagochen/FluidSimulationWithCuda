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

#define _CPU_ROUTINE_CPP_

#include <SGE\SGUtils.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace sge;

////////////////////////////////////////////////////////////////////////
///

std::vector<float*> devices;
cudaError cuda_status;

#define dev_u devices[0]
#define dev_v devices[1]
#define dev_pre_u devices[2]
#define dev_pre_v devices[3]
#define dev_dens devices[4]
#define dev_pre_dens devices[5];
#define dev_grid devices[6]
#define dev_pre_grid devices[7]

#define CALL_FUNC_FAILED(str) { ErrorMSG(str); return cuda_status; }

///
////////////////////////////////////////////////////////////////////////
///

cudaError CUDADevicesInit(void)
{
	// Choose which GPU to run on, change this on a multi-GPU system
	if ( (cuda_status = cudaSetDevice(0)) != cudaError::cudaSuccess )
		CALL_FUNC_FAILED("cudaSetDevice failed!");

	for (int i=0; i<8; i++)
	{
		float *ptr;

		if ( (cuda_status = cudaMalloc((void**)&ptr, ENTIRE_GRIDS_NUMBER * sizeof(float))) != cudaError::cudaSuccess )
			CALL_FUNC_FAILED("cudaMalloc failed!");

		devices.push_back(ptr);
	}
	
	return cuda_status;
};

cudaError CUDADevicesWait(void)
{
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch
	if ( (cuda_status = cudaDeviceSynchronize()) != cudaError::cudaSuccess)
		CALL_FUNC_FAILED("cudaDeviceSynchronize failed!");

	return cuda_status;
};

cudaError CUDADevicesEnds(void)
{
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces
	if ( (cuda_status = cudaDeviceReset()) != cudaError::cudaSuccess ) 
		CALL_FUNC_FAILED("cudaDeviceReset failed!");

	for (int i=0; i<6; i++)
	{
		if ( (cuda_status = cudaFree(devices[i])) != cudaError::cudaSuccess )
			CALL_FUNC_FAILED("cudaFree failed!");
	}

	devices.empty();

	return cuda_status;
};

///
////////////////////////////////////////////////////////////////////////
///

__global__ void add_source ( float *grid_out, const float *src_in, float dt_in )
{
	// Get the index of each thread
	int i = threadIdx.x;
	int j = threadIdx.y;
	int index = Index(i,j);
	
	// Append source to grid
	grid_out[index] += dt_in * src_in[index];
}

__global__ void set_bnd ( float *grid_out, int boundary )
{
#define is ==
	// Get the index of each thread
	int i = threadIdx.x;
	int j = threadIdx.y;

	// Boundary condition
	if ( i >= 1 && i <= GRIDS_WITHOUT_GHOST && j >= 1 && j <= GRIDS_WITHOUT_GHOST )
	{
		// Slove line (0, y)
		grid_out[Index(0, j)]  = boundary is 1 ? -grid_out[Index(1, j)] : grid_out[Index(1, j)];
		// Slove line (65, y)
		grid_out[Index(65, j)] = boundary is 1 ? -grid_out[Index(64,j)] : grid_out[Index(64,j)];
		// Slove line (x, 0)
		grid_out[Index(i, 0)]  = boundary is 2 ? -grid_out[Index(i, 1)] : grid_out[Index(i, 1)];
		// Slove line (x, 65)
		grid_out[Index(i, 65)] = boundary is 2 ? -grid_out[Index(i,64)] : grid_out[Index(i,64)];
	}
	// Slove ghost cell (0, 0)
	grid_out[Index(0, 0)] = 0.5f * ( grid_out[Index(1, 0)]  + grid_out[Index(0, 1)] );
	// Slove ghost cell (0, 65)
	grid_out[Index(0, 65)] = 0.5f * ( grid_out[Index(1, 65)] + grid_out[Index(0, 64)] );
	// Slove ghost cell (65, 0)
	grid_out[Index(65, 0)] = 0.5f * ( grid_out[Index(64, 0)] + grid_out[Index(65, 1)] );
	// Slove ghost cell (65, 65)
	grid_out[Index(65, 65)] = 0.5f * ( grid_out[Index(64, 65)] + grid_out[Index(65, 64)]);

#undef is
}


__global__ void lin_solve ( float *grid_inout, float *grid0_in, int boundary, float a, float c )
{
	// Get the index of each thread
	int i = threadIdx.x;
	int j = threadIdx.y;

	if ( i >= 1 && i <= GRIDS_WITHOUT_GHOST && j >= 1 && j <= GRIDS_WITHOUT_GHOST )
	{	
		grid_inout[Index(i,j)] = (grid0_in[Index(i,j)] + a*(grid_inout[Index(i-1,j)]+grid_inout[Index(i+1,j)]+grid_inout[Index(i,j-1)]+grid_inout[Index(i,j+1)]))/c;	
	}
}


cudaError diffuse ( int boundary, float *grid, float *grid0, float diff, float dt )
{
	float a = dt * diff * GRIDS_WITHOUT_GHOST * GRIDS_WITHOUT_GHOST;

	dim3 blocksize(ENTIRE_GRIDS_NUMBER, ENTIRE_GRIDS_NUMBER);

	// Copy data from host to GPU buffers
	cuda_status = cudaMemcpy(dev_grid, grid, ENTIRE_GRIDS_NUMBER * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (cuda_status != cudaError::cudaSuccess)
		CALL_FUNC_FAILED("cudaMemcpy failed!");
	cuda_status = cudaMemcpy(dev_pre_grid, grid0, ENTIRE_GRIDS_NUMBER * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (cuda_status != cudaError::cudaSuccess)
		CALL_FUNC_FAILED("cudaMemcpy failed!");

	// Launch kernel
	for ( int i = 0; i < 20; i++ )
		lin_solve <<<1, blocksize>>> ( dev_grid, dev_pre_grid, boundary, a, 1+4*a );
	set_bnd <<<1, blocksize>>> ( dev_grid, boundary );
}


__global__ void advect ( int GridSize, int boundary, float * density, float * density0, float * u, float * v, float dt )
{
	int i, j, i0, j0, i1, j1;
	float grid, y, s0, t0, s1, t1, dt0;

	dt0 = dt*GridSize;
	for ( i=1 ; i<=GridSize ; i++ ) 
	{
		for ( j=1 ; j<=GridSize ; j++ ) 
		{
			grid = i-dt0*u[Index(i,j)]; y = j-dt0*v[Index(i,j)];
			if (grid<0.5f) grid=0.5f; if (grid>GridSize+0.5f) grid=GridSize+0.5f; i0=(int)grid; i1=i0+1;
			if (y<0.5f) y=0.5f; if (y>GridSize+0.5f) y=GridSize+0.5f; j0=(int)y; j1=j0+1;
			s1 = grid-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
			density[Index(i,j)] = s0*(t0*density0[Index(i0,j0)]+t1*density0[Index(i0,j1)])+
				s1*(t0*density0[Index(i1,j0)]+t1*density0[Index(i1,j1)]);
		}
	}
	set_bnd ( GridSize, boundary, density );
}


__global__ void project ( int GridSize, float * u, float * v, float * p, float * div )
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

	lin_solve ( GridSize, 0, p, div, 1, 4 );

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

///
////////////////////////////////////////////////////////////////////////
///

#define interrupt return;

#if GPU_ON

void dens_step(float * grid, float * grid0, float * u, float * v)
{
	if (CUDADevicesInit() != cudaError::cudaSuccess)
	{
		ErrorMSG("Initialize CUDA devices failed!");
		interrupt;
	}

	if (CUDADevicesEnds() != cudaError::cudaSuccess)
	{
		ErrorMSG("Release CUDA devices failed!");
	}
};

void vel_step(float * u, float * v, float * u0, float * v0)
{
	if (CUDADevicesInit() != cudaError::cudaSuccess)
	{
		ErrorMSG("Initialize CUDA devices failed!");
		interrupt;
	}

	if (CUDADevicesEnds() != cudaError::cudaSuccess)
	{
		ErrorMSG("Release CUDA devices failed!");
		interrupt;
	}
};

#endif

///
////////////////////////////////////////////////////////////////////////