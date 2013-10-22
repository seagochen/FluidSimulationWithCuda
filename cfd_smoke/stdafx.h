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
* <First>       Oct 22, 2013
* <Last>		Oct 22, 2013
* <File>        stdafx.h
*/

#ifndef __stdafx_h_
#define __stdafx_h_

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Windows Header Files
  ----------------------------------------------------------------------
*/

#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "macro_definitions.h"
#include "cfd_visual.h"
#include "resource.h"

///
//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Undefine the following, if they were defined
  ----------------------------------------------------------------------
*/


#define index(i, j)  ((j) * Tile_X + (i))
#define cuda_device(gridDim, blockDim) <<<gridDim, blockDim>>>

///
//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Function Prototypes
  ----------------------------------------------------------------------
*/

extern void dens_step(float * grid, float * grid0, float * u, float * v);

extern void vel_step(float * u, float * v, float * u0, float * v0);

///
//////////////////////////////////////////////////////////////////////////////////////////

#endif