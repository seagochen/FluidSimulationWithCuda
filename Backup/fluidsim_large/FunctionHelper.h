/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 12, 2014
* <File Name>     FunctionHelper.h
*/


#ifndef __function_helper_h_
#define __function_helper_h_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <SGE\SGUtils.h>
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

		SGVOID DeviceDim1D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT gridx );
		SGVOID DeviceDim2D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT tile, SGINT gridx, SGINT gridy );
		SGVOID DeviceDim3D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT tile, SGINT gridx, SGINT gridy, SGINT gridz );
	};
};

#endif