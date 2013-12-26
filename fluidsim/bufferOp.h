#ifndef __buffer_op_h_
#define __buffer_op_h_

#include "fluidsim.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void kernelPickData ( unsigned char *data, int const *grid )
{
	GetIndex();

	int temp = grid [ Index (i, j, k) ];
	if ( temp > 256 ) temp = 256;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] = (unsigned char) temp;
};

__global__ void kernelPickData ( unsigned char *data, int const *grid1, int const *grid2, int const *grid3 )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	// Add data from grid 1
	int temp = grid1 [ Index (i, j, k) ];
	if ( temp > 256 ) temp = 256;
	else if ( temp < 0 ) temp = 0;
	data [ Index (i, j, k) ] += (unsigned char) temp;

	// Add data from grid 2
	temp = grid2 [ Index (i, j, k) ];
	if ( temp > 256 ) temp = 256;
	else if ( temp < 0 ) temp = 0;
	data [ Index (i, j, k) ] += (unsigned char) temp;
	
	// Add data from grid 2
	temp = grid3 [ Index (i, j, k) ];
	if ( temp > 256 ) temp = 256;
	else if ( temp < 0 ) temp = 0;
	data [ Index (i, j, k) ] += (unsigned char) temp;
};

#endif