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

	data [ Index (i, j, k) ] = 0;

	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] = (unsigned char) temp;
};

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