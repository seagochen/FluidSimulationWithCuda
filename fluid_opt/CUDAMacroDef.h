/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 25, 2014
* <Last Time>     Mar 01, 2014
* <File Name>     CUDAMacroDef.h
*/

#ifndef __cuda_macro_def_h_
#define __cuda_macro_def_h_

#include <stdio.h>
#include <cuda_runtime.h>
#include "DataStructures.h"

#define Index2D(i,j,num) ((j)*(num)+(i))
#define Index3D(i,j,k,num) ((k)*(num)*(num)+(j)*(num)+(i))
#define Index(i,j,k) Index3D(i,j,k,GRID_X)

#define cudaTrans2DTo3D(i,j,k,num) \
	k = Index2D((i),(j),(num)) / ((num)*(num)); \
	i = (i) % (num); \
	j = (j) % (num); \

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = TILE_X; \
	blockDim.y = TILE_X; \
	gridDim.x  = GRID_X / TILE_X; \
	gridDim.y  = GRID_X / TILE_X; \

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (GRID_X / TILE_X); \
	blockDim.y = (THREAD_X / TILE_X); \
	gridDim.x  = (GRID_X / blockDim.x); \
	gridDim.y  = (GRID_X * GRID_X * GRID_X) / (blockDim.x * blockDim.y * (GRID_X / blockDim.x)); \

#define GetIndex() \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D(i,j,k,GRID_X); \

#define gst_header        0            // (ghost, halo) the header cell of grid
#define sim_header        1            // (actually) the second cell of grid
#define gst_tailer       GRID_X - 1    // (ghost, halo) the last cell of grid
#define sim_tailer       GRID_X - 2    // (actually) the second last cell of grid

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_tailer and \
	j >= sim_header and j <= sim_tailer and \
	k >= sim_header and k <= sim_tailer ) {

#define EndSimArea() }

#endif