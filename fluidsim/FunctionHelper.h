/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Jan 31, 2014
* <File>        FunctionHelper.h
*/

#ifndef __function_helper_h_
#define __function_helper_h_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <string>
#include "DataStructures.h"

using std::string;

namespace sge
{
	class FuncHelper
	{
	public:
		SGVOID CheckErrors( const char* msg, const char *file, const int line );
		SGVOID DeviceDim2D( dim3 *grid_out, dim3 *block_out );
		SGVOID DeviceDim3D( dim3 *grid_out, dim3 *block_out );
		string string_fmt ( const std::string fmt_str, ... );
		SGRUNTIMEMSG PreBasicFluidKernel( SGINT nPtrs, SGDOUBLE **dStores, ... );
	};
}

#endif