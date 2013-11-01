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
* <Last>		Oct 25, 2013
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
   Definition for Variables of Computational Fluid Dynamics
  ----------------------------------------------------------------------
*/
#define Grids_X              32       // total grids number
#define SimArea_X            30       // simulation area which is the number without ghost cells

#define Tile_X               16       // 16x16 gpu threads as a block

#define DELTA_TIME           0.1f     // 0.1 second
#define DIFFUSION            0.0f     // diffusion rate
#define VISCOSITY            0.0f     // viscosity rate
#define FORCE                5.0f     // external force rate
#define SOURCE               100.0f   // to given a density with 100 percent

#define Client_X             512      // application's client size

#define BYTES_PER_TEXEL      3        // a pixel consists of three unsigned bytes

#define SIZE                 Grids_X * Grids_X

///
//////////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Data used in CFD
  ----------------------------------------------------------------------
*/

#ifdef __launch_main_cpp_

float * u, * v, * u_prev, * v_prev;
float * dens, * dens_prev;

std::vector<float*> dev_list;
sge::FileManager    Logfile;

cudaError cudaStatus;

#else

extern float * u, * v, * u_prev, * v_prev;
extern float * dens, * dens_prev;

extern std::vector<float*> dev_list;
extern sge::FileManager    Logfile;

extern cudaError cudaStatus;

#endif


/*
  ----------------------------------------------------------------------
   Function Definitions
  ----------------------------------------------------------------------
*/

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = Tile_X; \
	blockDim.y = Tile_X; \
	gridDim.x  = Grids_X / Tile_X; \
	gridDim.y  = Grids_X / Tile_X;

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (Grids_X / Tile_X); \
	blockDim.y = (256 / Tile_X); \
	gridDim.x  = (Grids_X / blockDim.x); \
	gridDim.y  = (Grids_X * Grids_X * Grids_X) / (blockDim.x * blockDim.y * (Grids_X / blockDim.x));

/*
  ----------------------------------------------------------------------
   external functions on CPU
  ----------------------------------------------------------------------
*/

extern void dens_step(float * grid, float * grid0, float * u, float * v);

extern void vel_step(float * u, float * v, float * u0, float * v0);


/*
  ----------------------------------------------------------------------
   etc.
  ----------------------------------------------------------------------
*/

#define devices   8

#define dev_u      dev_list[0]
#define dev_v      dev_list[1]
#define dev_u0     dev_list[2]
#define dev_v0     dev_list[3]
#define dev_den    dev_list[4]
#define dev_den0   dev_list[5]
#define dev_grid   dev_list[6]
#define dev_grid0  dev_list[7]

///
//////////////////////////////////////////////////////////////////////////////////////////////

#endif