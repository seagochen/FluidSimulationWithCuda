/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Jan 25, 2014
* <File>        FunctionHelper.h
*/

#ifndef __function_helper_h_
#define __function_helper_h_

#include <device_launch_parameters.h>
#include <string>

using std::string;

namespace sge
{
	class CUDAHelper
	{
	public:
		void CheckErrors( const char* msg, const char *file, const int line );
		void DeviceDim2D( dim3 *grid_out, dim3 *block_out );
		void DeviceDim3D( dim3 *grid_out, dim3 *block_out );
	};

	class AppHelper
	{
	public:
		string string_fmt ( const std::string fmt_str, ... );
	};
};

#endif