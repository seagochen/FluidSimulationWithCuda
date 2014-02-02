/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 01, 2014
* <File Name>     FunctionHelper.h
*/


#ifndef __function_helper_h_
#define __function_helper_h_

#include <device_launch_parameters.h>
#include "DataStructures.h"

namespace sge
{
	/* Function Helper，这里所定义的是在程序中常用到部分函数的集合 */
	class FunctionHelper
	{
	public:
		SGVOID CheckRuntimeErrors( const char* msg, const char *file, const int line );
		SGVOID DeviceDim2Dx( dim3 *grid_out, dim3 *block_out );
		SGVOID DeviceDim3Dx( dim3 *grid_out, dim3 *block_out );

		SGRUNTIMEMSG CreateDoubleBuffers( SGINT size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateIntegerBuffers( SGINT size, SGINT nPtrs, ... );

		void CopyData( SGDOUBLE *buffer, const SGDEVICEBUFF *devbuffs, SGFIELDTYPE type, SGNODECOORD coord );
		void CopyData( SGDEVICEBUFF *devbuffs, const SGDOUBLE *buffer, SGFIELDTYPE type, SGNODECOORD coord );
	};

};

#endif