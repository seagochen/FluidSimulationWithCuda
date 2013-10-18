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
* <Last>		Oct 18, 2013
* <File>        Macro_Definitions.h
*/

#ifndef _MACRO_DEFINITIONS_H_
#define _MACRO_DEFINITIONS_H_

//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Normal Definitions
  ----------------------------------------------------------------------
*/

#define True  1
#define False 0
/*
  ----------------------------------------------------------------------
   Definition for Variables of Computational Fluid Dynamics
  ----------------------------------------------------------------------
*/

#define CELLS_X              126      // grids number without ghost grids
#define GRIDS_X              128      // grids number contains ghost grids
#define TILES_X              16       // divide grids into 16 tiles
#define DELTA_TIME           0.1f     // 0.1 second
#define DIFFUSION            0.0f     // diffusion rate
#define VISCOSITY            0.0f     // viscosity rate
#define FORCE                5.0f     // external force rate
#define SOURCE               100.0f   // to given a density of 100 %
#define CLIENT_WIDTH         800      // application window size, width
#define CLIENT_HEIGHT        600      // application window size, height

/*
  ----------------------------------------------------------------------
   Definition for Visualization
  ----------------------------------------------------------------------
*/

#define BACK_COLOR_REDf      0.f      // background color, red component
#define BACK_COLOR_GREENf    0.1f     // background color, green component
#define BACK_COLOR_BLUEf     0.2f     // background color, blue component
#define BYTES_PER_TEXEL      3        // a pixel is composed of three components (rgb)

///
//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Function Definitions
  ----------------------------------------------------------------------
*/

#include "Macro_Functions.h"

/*
  ----------------------------------------------------------------------
   external functions on CPU
  ----------------------------------------------------------------------
*/


void dens_step(float * grid, float * grid0, float * u, float * v);

void vel_step(float * u, float * v, float * u0, float * v0);


///
//////////////////////////////////////////////////////////////////////////////////////////
///

#include <vector>
#include <cuda_runtime.h>

#ifdef _MAIN_CPP_

/*
  ----------------------------------------------------------------------
   Data used in CFD
  ----------------------------------------------------------------------
*/

int GridSize;
float dt, diff, visc;
float force, source;
float * u, * v, * u_prev, * v_prev;
float * dens, * dens_prev;
bool mouse_down[2];
int omx, omy, mx, my;
int win_x, win_y;

/*
  ----------------------------------------------------------------------
   Data used in CUDA
  ----------------------------------------------------------------------
*/

std::vector<float*> dev_list;
cudaError cudaStatus;

#else

/*
  ----------------------------------------------------------------------
   Data used in CFD
  ----------------------------------------------------------------------
*/

extern int GridSize;
extern float dt, diff, visc;
extern float force, source;
extern float * u, * v, * u_prev, * v_prev;
extern float * dens, * dens_prev;
extern bool mouse_down[2];
extern int omx, omy, mx, my;
extern int win_x, win_y;

/*
  ----------------------------------------------------------------------
   Data used in CUDA
  ----------------------------------------------------------------------
*/

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
//////////////////////////////////////////////////////////////////////////////////////////

#endif