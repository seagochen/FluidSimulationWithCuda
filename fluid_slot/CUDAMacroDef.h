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
#include "MacroDefinition.h"

#define Index2D(i,j,num) ((j)*(num)+(i))

#define Index3D(i,j,k,num) ((k)*(num)*(num)+(j)*(num)+(i))

#define cudaDeviceDim1D() \
	dim3 blockDim, gridDim; \
	blockDim.x = TPBUFFER_X; \
	blockDim.y = 1; \
	gridDim.x = 1; \
	gridDim.y = 1; \

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

#define GetIndex1D() \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \

#define GetIndex2D() \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \

#define GetIndex3D() \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	k = Index2D(i,j,GRID_X) / ((GRID_X)*(GRID_X)); \
	i = (i) % (GRID_X); \
	j = (j) % (GRID_X); \

#endif