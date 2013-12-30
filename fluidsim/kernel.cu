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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "fluidsim.h"
#include "bufferOp.h"
#include "myMath.h"

using namespace sge;
using namespace std;


__global__ void kernelAddSource ( double *dens, double *vel_u, double *vel_v, double *vel_w )
{
	GetIndex();

	if ( dens != NULL && j < 10 )
		dens [ Index(i,j,k) ] = INDENSITY;

	if ( vel_v != NULL && j < 10 )
		vel_v [ Index(i,j,k) ] = 1.f;
};

__global__ void kernelGridAdvection ( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();

	double u = i - u_in [ Index(i,j,k) ] * DELTA_TIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTA_TIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTA_TIME;

	grid_out [ Index(i,j,k) ] = trilinear ( grid_in, u, v, w );
};

void FluidSimProc::VelocitySolver ( void )
{
	cudaDeviceDim3D ();

	kernelAddSource <<<gridDim, blockDim>>> ( NULL, dev_u, dev_v, dev_w );
	if ( cudaThreadSynchronize() != cudaSuccess )  goto Error;

	kernelGridAdvection <<<gridDim, blockDim>>> ( dev_den0, dev_den, dev_u, dev_v, dev_w );
	if ( cudaThreadSynchronize() != cudaSuccess )  goto Error;

	kernelSwapBuffer <<<gridDim, blockDim>>> ( dev_den0, dev_den );
	if ( cudaThreadSynchronize() != cudaSuccess )  goto Error;

	goto Success;

Error:
	cudaCheckErrors ("cudaThreadSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	;
};

void FluidSimProc::DensitySolver ( void )
{
	cudaDeviceDim3D ();

	kernelAddSource <<<gridDim, blockDim>>> ( dev_den0, NULL, NULL, NULL );
	if ( cudaThreadSynchronize() != cudaSuccess )  goto Error;

	kernelSwapBuffer <<<gridDim, blockDim>>> ( dev_den0, dev_den );
	if ( cudaThreadSynchronize() != cudaSuccess )  goto Error;

	goto Success;

Error:
	cudaCheckErrors ("cudaThreadSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	;
};

void FluidSimProc::FluidSimSolver ( fluidsim *fluid )
{
	if ( !fluid->drawing.bContinue ) return ;

	// For fluid simulation, copy the data to device
	CopyDataToDevice();

	// Fluid process
	VelocitySolver ();
	DensitySolver ();
	PickData ( fluid );

	// Synchronize the device
	if ( cudaDeviceSynchronize() != cudaSuccess ) goto Error;

	// After simulation process, retrieve data back to host, in order to 
	// avoid data flipping
	CopyDataToHost();

	goto Success;

Error:
	cudaCheckErrors ("cudaDeviceSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	fluid->volume.ptrData = host_data;
};

void FluidSimProc::PickData ( fluidsim *fluid )
{
	cudaDeviceDim3D ();
	kernelPickData  <<<gridDim, blockDim>>> ( dev_data, dev_den );

	if ( cudaMemcpy (host_data, dev_data, 
		sizeof(unsigned char) * (fluid->volume.nVolDepth * fluid->volume.nVolHeight * fluid->volume.nVolWidth), 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed", __FILE__, __LINE__);
		FreeResourcePtrs ();
		exit (1);
	}
};