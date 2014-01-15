/**
* <Author>      Orlando Chen
* <First>       Oct 10, 2013
* <Last>		Dec 23, 2013
* <File>        Operation.h
*/

#ifndef __operation_h_
#define __operation_h_

#include "FluidSimArea.h"
#include "FluidMathLib.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernelPickData ( unsigned char *data, double const *grid )
{
	GetIndex();

	// Zero data first
	data [ Index (i, j, k) ] = 0;

	// retrieve data from grid
	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 0 and temp < 250 )
		data [ Index(i, j, k) ] = (unsigned char) temp;
};

/*
__global__ void kernelPickData ( unsigned char *data, double const *grid1, double const *grid2, double const *grid3 )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	// Add data from grid 1
	int temp = sground ( grid1[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	// Add data from grid 2
	temp += sground ( grid2[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;
	
	// Add data from grid 2
	temp += sground ( grid3[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] += (unsigned char) temp;
};
*/

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