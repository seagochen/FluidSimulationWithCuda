/**
* <Author>      Orlando Chen
* <First>       Oct 10, 2013
* <Last>		Jan 12, 2014
* <File>        BufferOperationDynamic.h
*/

#ifndef __buffer_operation_dynamic_h_
#define __buffer_operation_dynamic_h_

#include "FluidSimAreaDynamic.h"
#include "FluidMathLibDynamic.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernelPickData ( unsigned char *data, double const *grid )
{
	GetIndex();

	/* zero data first */
	data [ Index (i, j, k) ] = 0;

	/* retrieve data from grid */
	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 0 and temp < 250 )
		data [ Index(i, j, k) ] = (unsigned char) temp;
};

__global__ void kernelCopyBuffer ( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ void kernelSwapBuffer ( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__global__ void kernelZeroBuffer ( double *grid )
{
	GetIndex ();
	grid [ Index(i,j,k) ] = 0.f;
};

__host__ void hostSwapBuffer ( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
};

#endif