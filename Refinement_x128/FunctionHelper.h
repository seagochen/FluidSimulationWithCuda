/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 26, 2014
* <File Name>     FunctionHelper.h
*/


#ifndef __function_helper_h_
#define __function_helper_h_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "MacroDefinition.h"

namespace sge
{
	class FunctionHelper
	{
	public:
		SGBOOLEAN GetCUDALastError( const char* msg, const char *file, const int line );

		SGRUNTIMEMSG CreateHostBuffers( size_t size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateDeviceBuffers( size_t size, SGINT nPtrs, ... );

		SGVOID FreeHostBuffers( SGINT nPtrs, ... );
		SGVOID FreeDeviceBuffers( SGINT nPtrs, ... );

		SGVOID DeviceParamDim( dim3 *gridDim, dim3 *blockDim, SGINT thread, SGINT tile );
		SGVOID DeviceParamDim( dim3 *gridDim, dim3 *blockDim, SGINT thread, SGINT tilex, SGINT tiley, SGINT gridx, SGINT gridy );
		SGVOID DeviceParamDim( dim3 *gridDim, dim3 *blockDim, SGINT thread, SGINT tilex, SGINT tiley, SGINT gridx, SGINT gridy, SGINT gridz );

#if 0
		SGVOID DeviceDim1D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT gridx );
		SGVOID DeviceDim2D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT tile, SGINT gridx, SGINT gridy );
		SGVOID DeviceDim3D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT tile, SGINT gridx, SGINT gridy, SGINT gridz );
#endif

		SGBOOLEAN CreateCompNodesForDevice( std::vector<double*> *vectBuf, size_t size, size_t nodes );
		SGBOOLEAN CreateCompNodesForHost( std::vector<double*> *vectBuf, size_t size, size_t nodes );
		SGBOOLEAN CreateCompNodesForDevice( std::vector<int*> *vectBuf, size_t size, size_t nodes );
		SGBOOLEAN CreateCompNodesForHost( std::vector<int*> *vectBuf, size_t size, size_t nodes );
		SGBOOLEAN CreateCompNodesForDevice( std::vector<unsigned char*> *vectBuf, size_t size, size_t nodes );
		SGBOOLEAN CreateCompNodesForHost( std::vector<unsigned char*> *vectBuf, size_t size, size_t nodes );
	};
};

#endif