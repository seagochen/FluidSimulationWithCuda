/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 25, 2014
* <Last Time>     Feb 02, 2014
* <File Name>     CUDAMacroDef.h
*/

#ifndef __cuda_macro_def_h_
#define __cuda_macro_def_h_

#include <stdio.h>
#include <cuda_runtime.h>
#include "DataStructures.h"

#define cudaDevice(gridDim,blockDim) <<<gridDim,blockDim>>>

#define cudaIndex2D(i,j,elements_x) ((j)*(elements_x)+(i))

#define cudaIndex3D(i,j,k,elements_x) ((k)*elements_x*elements_x+(j)*elements_x+(i))

#define Index(i,j,k) cudaIndex3D(i,j,k,GRIDS_X)

#define cudaTrans2DTo3D(i,j,k,elements_x) \
	k = cudaIndex2D(i,j,(elements_x)) / ((elements_x)*(elements_x)); \
	i = i % (elements_x); \
	j = j % (elements_x); \

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = TILE_X; \
	blockDim.y = TILE_X; \
	gridDim.x  = GRIDS_X / TILE_X; \
	gridDim.y  = GRIDS_X / TILE_X; \

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (GRIDS_X / TILE_X); \
	blockDim.y = (THREADS_X / TILE_X); \
	gridDim.x  = (GRIDS_X / blockDim.x); \
	gridDim.y  = (GRIDS_X * GRIDS_X * GRIDS_X) / (blockDim.x * blockDim.y * (GRIDS_X / blockDim.x)); \

#define GetIndex()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D ( i, j, k, GRIDS_X ); \

#define gst_header        0             /* (ghost, halo) the header cell of grid */
#define sim_header        1             /* (actually) the second cell of grid */
#define gst_tailer       GRIDS_X - 1    /* (ghost, halo) the last cell of grid */
#define sim_tailer       GRIDS_X - 2    /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_tailer ) \
	if ( j >= sim_header and j <= sim_tailer ) \
	if ( k >= sim_header and k <= sim_tailer ) {

#define EndSimArea() }

#endif