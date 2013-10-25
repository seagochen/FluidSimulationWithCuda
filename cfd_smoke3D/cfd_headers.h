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
* <First>       Oct 25, 2013
* <Last>		Oct 25, 2013
* <File>        cfd_headers.h
*/

#ifndef __cfd_headers_h_
#define __cfd_headers_h_

//////////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   C Standard & STL Headers and System Headers
  ----------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

/*
  ----------------------------------------------------------------------
   OpenGL Headers
  ----------------------------------------------------------------------
*/

#include <GL\glew.h>
#include <GL\glut.h>

/*
  ----------------------------------------------------------------------
   CUDA Headers
  ----------------------------------------------------------------------
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
  ----------------------------------------------------------------------
   Intel MKL Headers
  ----------------------------------------------------------------------
*/

/*
  ----------------------------------------------------------------------
   SGEngine Headers
  ----------------------------------------------------------------------
*/

#include <SGE\SGUtils.h>

/*
  ----------------------------------------------------------------------
   etc.
  ----------------------------------------------------------------------
*/

///
//////////////////////////////////////////////////////////////////////////////////////////////

#endif