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
* <Last>		Oct 20, 2013
* <File>        cfd_kernel.h
*/

#ifndef __cfd_kernel_h_
#define __cfd_kernel_h_

//////////////////////////////////////////////////////////////////////////////////////////
///

#include "macro_definitions.h"

/*
  ----------------------------------------------------------------------
   Define dll interfaces
  ----------------------------------------------------------------------
*/
#ifdef _In_Dll_File
#define _DLL __declspec(dllexport)    // in dll sources, function prefix
#else
#define _DLL __declspec(dllimport)    // Headers included from external  
#endif


/*
  ----------------------------------------------------------------------
   external functions on CPU
  ----------------------------------------------------------------------
*/


_DLL void dens_step(float * grid, float * grid0, float * u, float * v);

_DLL void vel_step(float * u, float * v, float * u0, float * v0);


#ifndef _In_Dll_File
#pragma comment(lib, "cfd_kernel.lib")
#endif

///
//////////////////////////////////////////////////////////////////////////////////////////

#endif