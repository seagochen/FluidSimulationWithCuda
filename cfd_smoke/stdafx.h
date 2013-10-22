// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Windows Header Files
  ----------------------------------------------------------------------
*/

#define _In_Dll_File

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cfd_kernel.h"

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