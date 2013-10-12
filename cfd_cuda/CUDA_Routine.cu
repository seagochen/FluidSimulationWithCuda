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
* <First>       Oct 12, 2013
* <Last>		Oct 12, 2013
* <File>        CUDA_Routine.cpp
*/

#define _CPU_ROUTINE_CPP_

#include "Macro_Definitions.h"
#include <cuda_runtime.h>
#include <SGE\SGUtils.h>
#include <vector>

using namespace sge;

////////////////////////////////////////////////////////////////////////
///

std::vector<float*> devices;

#define dev_u devices[0]
#define dev_v devices[1]
#define dev_pre_u devices[2]
#define dev_pre_v devices[3]
#define dev_dens devices[4]
#define dev_pre_dens devices[5];

#define CALL_FUNC_FAILED(str) { ErrorMSG(str); return SGRUNTIMEMSG::SG_CALL_EXTFUNC_FAILED_EXCEP; }

///
////////////////////////////////////////////////////////////////////////
///

SGRUNTIMEMSG CUDADevicesInit(void)
{
	// Choose which GPU to run on, change this on a multi-GPU system
	if (cudaSetDevice(0) != cudaError::cudaSuccess)
		CALL_FUNC_FAILED("cudaSetDevice failed!");

	for (int i=0; i<6; i++)
	{
		float *ptr;

		if (cudaMalloc((void**)&ptr, ENTIRE_GRIDS_NUMBER * sizeof(float)) != cudaError::cudaSuccess)
			CALL_FUNC_FAILED("cudaMalloc failed!");

		devices.push_back(ptr);
	}
	
	return SGRUNTIMEMSG::SG_RUNTIME_OK;
};

SGRUNTIMEMSG CUDADevicesWait(void)
{
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch
	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
		CALL_FUNC_FAILED("cudaDeviceSynchronize failed!");
};

SGRUNTIMEMSG CUDADevicesEnds(void)
{
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces
	if (cudaDeviceReset() != cudaError::cudaSuccess) 
		CALL_FUNC_FAILED("cudaDeviceReset failed!");

	for (int i=0; i<6; i++)
	{
		if (cudaFree(devices[i]) != cudaError::cudaSuccess)
			CALL_FUNC_FAILED("cudaFree failed!");
	}

	devices.empty();

	return SGRUNTIMEMSG::SG_RUNTIME_OK;
};

///
////////////////////////////////////////////////////////////////////////
///


///
////////////////////////////////////////////////////////////////////////
///

#if GPU_ON

void dens_step(float * grid, float * grid0, float * u, float * v);

void vel_step(float * u, float * v, float * u0, float * v0);

#endif

///
////////////////////////////////////////////////////////////////////////