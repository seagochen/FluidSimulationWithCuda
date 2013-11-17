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
* <First>       Oct 7, 2013
* <Last>		Nov 6, 2013
* <File>        macroDef.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#include "cfdHeaders.h"
#include "cudaHelper.h"

//////////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Parameters Needed for launching CUDA Device
  ----------------------------------------------------------------------
*/

#define Grids_X              128 /* --------------------------------------- grids number on each dimension */
#define SimArea_X            126 /* --------------------------------------- number of grids without ghost cells */
#define Tile_X               16  /* --------------------------------------- ties 16x16 gpu threads as a block */
#define SIM_SIZE             Grids_X * Grids_X * Grids_X /* --------------- 3-D grid for CFD */

/*
  ----------------------------------------------------------------------
   Parameters Needed for Computational Fluid Dynamics
  ----------------------------------------------------------------------
*/

#define DELTA_TIME           0.1f /* -------------------------------------- 0.1 second */
#define DIFFUSION            0.0f /* -------------------------------------- diffusion rate */
#define VISCOSITY            0.0f /* -------------------------------------- viscosity rate */
#define FORCE                5.0f /* -------------------------------------- external force rate */
#define SOURCE               100.0f /* ------------------------------------ to given a density with 100 percent */

/*
  ----------------------------------------------------------------------
   Parameters Needed for Rendering
  ----------------------------------------------------------------------
*/

#define DIS_SIZE             Grids_X * Grids_X /*-------------------------- rendering buffer size */
#define Client_X             512 /* --------------------------------------- client size */

///
//////////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Data used in CFD
  ----------------------------------------------------------------------
*/

#ifdef __launch_main_cpp_

std::vector<float*> dev_list;
std::vector<float*> host_list;
std::vector<float*> buff_list;
sge::FileManager    Logfile;
cudaError cudaStatus;

#else

extern std::vector<float*> dev_list;
extern std::vector<float*> host_list;
extern std::vector<float*> buff_list;
extern sge::FileManager    Logfile;
extern cudaError cudaStatus;

#endif


/*
  ----------------------------------------------------------------------
   Function Definitions
  ----------------------------------------------------------------------
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


/*
  ----------------------------------------------------------------------
   external functions on CPU
  ----------------------------------------------------------------------
*/

extern void DensitySolver(float *grid, float *grid0, float *u, float *v, float *w);

extern void VelocitySolver(float *u, float *v, float *w, float *u0, float *v0, float *w0);

extern void DensityInterpolate ( void );

extern void VelocityInterpolate ( void );

extern void cudaCheckRuntimeErrors ( char *msg );

/*
  ----------------------------------------------------------------------
   etc.
  ----------------------------------------------------------------------
*/


#define DevListNum    10

#define dev_u         dev_list [ 0 ] /* ----- u of U, on CUDA */
#define dev_v         dev_list [ 1 ] /* ----- v of U, on CUDA */
#define dev_w         dev_list [ 2 ] /* ----- w of U, on CUDA */
#define dev_u0        dev_list [ 3 ] /* ----- original u of U, on CUDA */
#define dev_v0        dev_list [ 4 ] /* ----- original v of U, on CUDA */
#define dev_w0        dev_list [ 5 ] /* ----- original w of U, on CUDA */
#define dev_den       dev_list [ 6 ] /* ----- density, on CUDA */
#define dev_den0      dev_list [ 7 ] /* ----- original density, on CUDA */
#define dev_grid      dev_list [ 8 ] /* ----- temporary grid, on CUDA */
#define dev_grid0     dev_list [ 9 ] /* ----- original temporary grid, on CUDA */

#define HostListNum   8

#define host_u        host_list [ 0 ] /* ----- component u of flow U on x-axis (left to right) */
#define host_v        host_list [ 1 ] /* ----- component v of flow U on y-axis (up to down) */
#define host_w        host_list [ 2 ] /* ----- component w of flow U on z-axis (near to far) */
#define host_u0       host_list [ 3 ] /* ----- original u */
#define host_v0       host_list [ 4 ] /* ----- original v */
#define host_w0       host_list [ 5 ] /* ----- original w */
#define host_den      host_list [ 6 ] /* ----- scalar field of density */
#define host_den0     host_list [ 7 ] /* ----- original density */

#define BufferListNum 5

#define host_disu     buff_list [ 0 ] /* ----- rendering buffer for u */
#define host_disv     buff_list [ 1 ] /* ----- rendering buffer for v */
#define host_disw     buff_list [ 2 ] /* ----- rendering buffer for w */
#define host_disD     buff_list [ 3 ] /* ----- rendering buffer for density */
#define dev_2DRender  buff_list [ 4 ] /* ----- ray casting buffer, on CUDA */

///
//////////////////////////////////////////////////////////////////////////////////////////////

#endif