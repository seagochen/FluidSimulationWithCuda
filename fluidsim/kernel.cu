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
* <First>       Dec 12, 2013
* <Last>		Dec 23, 2013
* <File>        kernel.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda.h>
#include <device_launch_parameters.h>
#include "fluidsim.h"
#include "bufferOp.h"
#include "myMath.h"

using namespace sge;
using namespace std;


__global__ void kernelAddSource ( int *dens, int *vel_u, int *vel_v, int *vel_w )
{
	GetIndex();

	vel_u [ Index(i,j,k) ] = i;
	vel_v [ Index(i,j,k) ] = j;
	vel_w [ Index(i,j,k) ] = k;
};


void FluidSimProc::FluidSimSolver ( fluidsim *fluid )
{
	if ( !fluid->bContinue )
		return ;

	cudaDeviceDim3D();

	kernelAddSource <<< gridDim, blockDim >>> ( NULL, dev_u, dev_v, dev_w );
	kernelPickData  <<< gridDim, blockDim >>> ( dev_data, dev_v );

	if ( cudaMemcpy (host_data, dev_data, 
		sizeof(unsigned char) * (fluid->nVolDepth * fluid->nVolHeight * fluid->nVolWidth), 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed");
		FreeResourcePtrs ();
		exit (1);
	};

	fluid->ptrData = host_data;
};