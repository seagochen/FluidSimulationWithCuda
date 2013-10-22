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
#include "cfd_kernel.h"
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

#ifdef index(i, j)
#undef index(i, j)
#endif

#define index(i, j)  ((j) * Tile_X + (i))

#ifdef cuda_device(gridDim, blockDim) 
#undef cuda_device(gridDim, blockDim) 
#endif

#define cuda_device(gridDim, blockDim) <<<gridDim, blockDim>>>

#ifdef swap(a, b)
#undef swap(a, b)
#endif

template <class T> void SWAP(T& a, T& b)
{
  T c(a); a=b; b=c;
}

#define swap(a, b) SWAP((a), (b))

///
//////////////////////////////////////////////////////////////////////////////////////////

#endif