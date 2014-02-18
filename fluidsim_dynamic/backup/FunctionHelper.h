/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 17, 2014
* <File Name>     FunctionHelper.h
*/


#ifndef __function_helper_h_
#define __function_helper_h_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "DataStructures.h"

namespace sge
{
	/* Function Helper，这里所定义的是在程序中常用到部分函数的集合 */
	class FunctionHelper
	{
	public:
		SGVOID CheckRuntimeErrors( const char* msg, const char *file, const int line );

		SGRUNTIMEMSG CreateDoubleBuffers( size_t size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateIntegerBuffers( size_t size, SGINT nPtrs, ... );

		SGRUNTIMEMSG CreateHostBuffers( size_t size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateDeviceBuffers( size_t size, SGINT nPtrs, ... );

		SGRUNTIMEMSG CreateVolumetricBuffers( size_t size, SGUCHAR **host_buf, SGUCHAR **dev_buf );
	};
};

#endif