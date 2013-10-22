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
* <Last>		Oct 22, 2013
* <File>        macro_definitions.h
*/

#ifndef _MACRO_DEFINITIONS_H_
#define _MACRO_DEFINITIONS_H_

#include <vector>

#pragma once

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
   Definition for Computational Fluid Dynamics
  ----------------------------------------------------------------------
*/

#define SimArea_X            62                       // grid number on single direction that without ghost cells 
#define Grids_X              64                       // grid number on single direction that contains ghost cells for boundary conditions
#define Tile_X               Grids_X / 16             // we divide each 16 cells in a block on single direction that means 2-D block contained 16x16 cells.
#define ClientSize_X         512                      // client size. This can be assigned to window width and height.
#define StepLenght_X         ClientSize_X / Grids_X   // rendering the result. this definition determines basic primitive size

#define DELTA_TIME           0.1f                     // 0.1 second
#define DIFFUSION            0.0f                     // diffusion rate
#define VISCOSITY            0.0f                     // viscosity rate
#define FORCE                5.0f                     // external force rate
#define SOURCE               100.0f                   // to given a density of 100 %

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

#ifdef __main_cpp_

std::vector<float*> ptr_list;

#else

extern std::vector<float*> ptr_list;

#endif

#define dev_u      ptr_list[0]
#define dev_v      ptr_list[1]
#define dev_u0     ptr_list[2]
#define dev_v0     ptr_list[3]
#define dev_den    ptr_list[4]
#define dev_den0   ptr_list[5]
#define dev_grid   ptr_list[6]
#define dev_grid0  ptr_list[7]
#define host_u     ptr_list[8]
#define host_v     ptr_list[9]
#define host_u0    ptr_list[10]
#define host_v0    ptr_list[11]
#define host_den   ptr_list[12]
#define host_den0  ptr_list[13]

#define dev_num      8
#define host_num     6
#define ptr_num      14

///
//////////////////////////////////////////////////////////////////////////////////////////

#endif