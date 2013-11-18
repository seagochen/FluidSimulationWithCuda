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
* <First>       Nov 18, 2013
* <Last>		Nov 18, 2013
* <File>        cudaHelper.h
*/

#ifndef __cuda_new_helper_h_
#define __cuda_new_helper_h_

/*
  -----------------------------------------------------------------------------------------------------------
   Custom CUDA Functions
  -----------------------------------------------------------------------------------------------------------
*/

#define cudaCheckRuntimeErrors(msg) \
	do { \
	extern void FreeResourcePtrs (void); \
	Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, \
		"%s, at line: %d of file %s", msg, __LINE__, __FILE__ ); \
	Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, \
		">>>> Error Message: %s", cudaGetErrorString ( cudaGetLastError( ) ) ); \
	FreeResourcePtrs ( ); exit ( 0 ); \
	} while(0); \

#define cudaDevice(gridDim, blockDim) <<<gridDim, blockDim>>>

#define cudaIndex2D(i, j, elements_x) ((j) * (elements_x) + (i))

#define cudaTrans2DTo3D(i, j, k, elements_x) { \
	k = cudaIndex2D(i, j, elements_x) / ((elements_x) * (elements_x)) ; \
	i = i % elements_x; \
	j = j % elements_x; \
	}

#define cudaIndex3D(i, j, k, elements_x) ((k) * elements_x * elements_x + (j) * elements_x + (i))

/*
  -----------------------------------------------------------------------------------------------------------
   The Following Variables will Determine the Way to use The CUDA Device
  -----------------------------------------------------------------------------------------------------------
*/

#define Grids_X              128 /* ----------------------------------- grids number on each dimension */
#define SimArea_X            126 /* ----------------------------------- number of grids without ghost cells */
#define Threads_X            512 /* ----------------------------------- number of threads enabled */
#define Tile_X               16  /* ----------------------------------- ties 16x16 gpu threads as a block */
#define SIM_SIZE             Grids_X * Grids_X * Grids_X /* ----------- CFD dimension of grid */

/*
  -----------------------------------------------------------------------------------------------------------
   Parameters Needed for Rendering
  -----------------------------------------------------------------------------------------------------------
*/

#define DIS_SIZE             Grids_X * Grids_X /*----------------------- buffer size for rendering */
#define Client_X             512 /* ------------------------------------ client size, unit: pixel */


/*
  -----------------------------------------------------------------------------------------------------------
   Function Definitions
  -----------------------------------------------------------------------------------------------------------
*/

#define Index(i, j, k)      cudaIndex3D(i, j, k, Grids_X)

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = Tile_X; \
	blockDim.y = Tile_X; \
	gridDim.x  = Grids_X / Tile_X; \
	gridDim.y  = Grids_X / Tile_X;

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (Grids_X / Tile_X); \
	blockDim.y = (512 / Tile_X); \
	gridDim.x  = (Grids_X / blockDim.x); \
	gridDim.y  = (Grids_X * Grids_X * Grids_X) / (blockDim.x * blockDim.y * (Grids_X / blockDim.x));

#define GetIndex()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D ( i, j, k, Grids_X );

#endif