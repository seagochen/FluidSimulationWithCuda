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
* <Last>		Dec 21, 2013
* <File>        kernel.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SGE\SGUtils.h>
#include "fluidsim.h"

using namespace sge;
using namespace std;

#pragma region free, zero, and allocate memory

void FluidSim::FreeResourcePtrs ( void )
{
	// Release ptr of host
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( host_list [ i ] ) SAFE_FREE_PTR ( host_list [ i ] );
	}
	host_list.empty ( );

	// Release ptr of CUDA
	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );
}

void FluidSim::ZeroData ( void )
{
	for ( int i = 0; i < SIM_SIZE; i++ )
	{
		host_u [ i ] = 0.f;
		host_v [ i ] = 0.f;
		host_w [ i ] = 0.f;
		host_u0 [ i ] = 0.f;
		host_v0 [ i ] = 0.f;
		host_w0 [ i ] = 0.f;
		host_den [ i ] = 0.f;
		host_den0 [ i ] = 0.f;
	}
}

SGRUNTIMEMSG FluidSim::AllocateResourcePtrs ( void )
{
	/// Choose which GPU to run on, change this on a multi-GPU system. ///
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices" );

	/// Allocate memory on host ///
	for ( int i = 0; i < HostListNum; i++ )
	{
		static float *ptr;
		ptr = (float*) malloc ( SIM_SIZE * sizeof(float) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
			return SG_RUNTIME_FALSE;
		}
	}

	/// Allocate memory on GPU devices ///
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static float *ptr;
		if ( cudaMalloc( (void **) &ptr, SIM_SIZE * sizeof(float) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	// Finally
	return SG_RUNTIME_OK;
}  

#pragma endregion


#pragma region kernels for add source and velocity

__global__ void kernelAddSource ( float *grid )
{
	GetIndex();

	const int half = Grids_X / 2;

	if ( i > half - 10 && i < half + 10 ) if ( j < 5 ) if ( k > half - 10 && k < half + 10 )
	{
		grid [ Index(i,j,k) ] = SOURCE;
	}
};

__global__ void kernelAddVelocity ( float *devU, float *devV, float *devW )
{
	GetIndex();
	
	const int half = Grids_X / 2;

	if ( i > half - 10 && i < half + 10 ) if ( j < 5 ) if ( k > half - 10 && k < half + 10 )
	{
		devV [ Index(i,j,k) ] = 0.01f;
	}
};

#pragma endregion


#pragma region kernels for boundary condition

__device__ void subkelCheckIntensity ( float *dens )
{
	// Get index of GPU-thread
	GetIndex ();

	// Boundary condition
	BeginSimArea ();
	{

	}
	EndSimArea ();
};

__global__ void kernelCheckBoundary ( float *dens, float *velU, float *velV, float *velW )
{
	subkelCheckIntensity ( dens );
}

#pragma endregion



#pragma region kernels for advect source and velocity



#pragma endregion


#pragma region retrieve data from buffer

__global__ void kernelPickData ( float *grid, unsigned char *data )
{
	GetIndex();

	float temp = ceil ( grid [ Index (i, j, k) ] );
	if ( temp > 256.f ) temp = 256.f;
	else if ( temp < 0.f ) temp = 0.f;

	unsigned char value = (unsigned char) temp;

	data [ Index (i, j, k) ] = value;
};

#pragma endregion


FluidSim::FluidSim ( fluidsim *fluid )
{
#pragma region allocate host memory and device buffer
	size_t size = fluid->nVolDepth * fluid->nVolHeight * fluid->nVolWidth;
	data = (GLubyte *) malloc ( size * sizeof(GLubyte) );
	if ( cudaMalloc( (void**) &dev_data, SIM_SIZE * sizeof(unsigned char) ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!" );
		cudaFree (dev_data);
		SAFE_FREE_PTR ( data );
		exit (1);
	}
	if ( AllocateResourcePtrs () != SG_RUNTIME_OK )
	{
		FreeResourcePtrs ();
		SAFE_FREE_PTR ( data );
		exit (1);
	}
#pragma endregion
	
	first = false;

	cout << "fluid simulation ready, zero the data and prepare the stage now" << endl;
	ZeroData ();
	
#pragma region do add source and velocity first
	cudaDeviceDim3D ();
	
	kernelAddSource cudaDevice ( gridDim, blockDim ) ( dev_den );
	kernelAddVelocity cudaDevice ( gridDim, blockDim ) ( dev_u, dev_v, dev_w );
	
	if ( cudaDeviceSynchronize () != cudaSuccess )
		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );

	/// We do not retrieve the data right now, just keep them on buffer for next step
#pragma endregion
};


void FluidSim::FluidSimSolver ( fluidsim *fluid )
{
	// Define the computing unit size
	cudaDeviceDim3D ();

	kernelPickData cudaDevice ( gridDim, blockDim ) ( dev_den, dev_data );

	if ( cudaDeviceSynchronize () != cudaSuccess )
		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );
	
	if ( cudaMemcpy ( data, dev_data, sizeof(unsigned char) * SIM_SIZE, cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );

	fluid->ptrData = data;
};