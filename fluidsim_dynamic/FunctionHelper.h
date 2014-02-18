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
		/* check CUDA runtime errors */
		SGVOID CheckRuntimeErrors( const char* msg, const char *file, const int line );

		/* malloc buffers for paticular data type */
		SGRUNTIMEMSG CreateHostCharBuffers( size_t size, SGINT nPtrs, ... );     // obsoleted
		SGRUNTIMEMSG CreateHostIntegerBuffers( size_t size, SGINT nPtrs, ... );  // obsoleted
		SGRUNTIMEMSG CreateHostDoubleBuffers( size_t size, SGINT nPtrs, ... );   // obsoleted

		SGRUNTIMEMSG CreateDeviceCharBuffers( size_t size, SGINT nPtrs, ... );    // obsoleted
		SGRUNTIMEMSG CreateDeviceIntegerBuffers( size_t size, SGINT nPtrs, ... ); // obsoleted
		SGRUNTIMEMSG CreateDeviceDoubleBuffers( size_t size, SGINT nPtrs, ... );// obsoleted

		SGRUNTIMEMSG CreateHostBuffers( size_t size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateDeviceBuffers( size_t size, SGINT nPtrs, ... );

		/* free buffers */
		SGVOID FreeHostCharBuffers( SGINT nPtrs, ... );    // obsoleted
		SGVOID FreeHostIntegerBuffers( SGINT nPtrs, ... ); // obsoleted
		SGVOID FreeHostDoubleBuffers( SGINT nPtrs, ... );  // obsoleted

		SGVOID FreeDeviceCharBuffers( SGINT nPtrs, ... );    // obsoleted
		SGVOID FreeDeviceIntegerBuffers( SGINT nPtrs, ... ); // obsoleted
		SGVOID FreeDeviceDoubleBuffers( SGINT nPtrs, ... );  // obsoleted

		SGVOID FreeHostBuffers( SGINT nPtrs, ... );
		SGVOID FreeDeviceBuffers( SGINT nPtrs, ... );
	};
};

#endif