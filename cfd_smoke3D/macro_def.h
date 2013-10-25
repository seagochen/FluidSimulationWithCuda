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
* <File>        macro_def.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Definition for Variables of Computational Fluid Dynamics
  ----------------------------------------------------------------------
*/
#define Grids_X              64       // total grids number
#define SimArea_X            62       // simulation area which is the number without ghost cells

#define Tile_X               16       // 16x16 gpu threads as a block

#define DELTA_TIME           0.1f     // 0.1 second
#define DIFFUSION            0.0f     // diffusion rate
#define VISCOSITY            0.0f     // viscosity rate
#define FORCE                5.0f     // external force rate
#define SOURCE               100.0f   // to given a density with 100 percent

#define Client_X             512      // application's client size

/*
  ----------------------------------------------------------------------
   Definition for Visualization
  ----------------------------------------------------------------------
*/

#define BYTES_PER_TEXEL 3

///
////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Function Definitions
  ----------------------------------------------------------------------
*/

#define PrintStatus(str) {system("cls"); printf("%s");}

#define Index(i,j)      ((j) * Grids_X + i)

#define GPUIndex(i, j)  ((j) * gridDim.x * blockDim.x + (i))

#define cuda_device(gridDim, blockDim) <<<gridDim, blockDim>>>

/*
  ----------------------------------------------------------------------
   external functions on CPU
  ----------------------------------------------------------------------
*/


void dens_step(float * grid, float * grid0, float * u, float * v);

void vel_step(float * u, float * v, float * u0, float * v0);


///
////////////////////////////////////////////////////////////////////////
///

#include <vector>
#include <cuda_runtime.h>

/*
  ----------------------------------------------------------------------
   Data used in CFD
  ----------------------------------------------------------------------
*/

#ifdef __launch_main_cpp_


float * u, * v, * u_prev, * v_prev;
float * dens, * dens_prev;

std::vector<float*> dev_list;
cudaError cudaStatus;

#else

extern float * u, * v, * u_prev, * v_prev;
extern float * dens, * dens_prev;

extern std::vector<float*> dev_list;
extern cudaError cudaStatus;

#endif

#define dev_u      dev_list[0]
#define dev_v      dev_list[1]
#define dev_u0     dev_list[2]
#define dev_v0     dev_list[3]
#define dev_den    dev_list[4]
#define dev_den0   dev_list[5]
#define dev_grid   dev_list[6]
#define dev_grid0  dev_list[7]

#define devices   8

///
////////////////////////////////////////////////////////////////////////


#endif