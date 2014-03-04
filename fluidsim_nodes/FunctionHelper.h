/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 03, 2014
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
	/* Function Helper，这里所定义的是在程序中常用到部分函数的集合 */
	class FunctionHelper
	{
	public:
		/* check CUDA runtime errors */
		SGBOOLEAN GetCUDALastError( const char* msg, const char *file, const int line );

		/* malloc buffers for paticular data type */
		SGRUNTIMEMSG CreateHostBuffers( size_t size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateDeviceBuffers( size_t size, SGINT nPtrs, ... );

		/* free buffers */
		SGVOID FreeHostBuffers( SGINT nPtrs, ... );
		SGVOID FreeDeviceBuffers( SGINT nPtrs, ... );
	};
};

#endif