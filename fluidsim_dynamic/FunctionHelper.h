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
#include "DataStructures.h"

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
}


#ifdef __cplusplus
extern "C"
{
#endif
	extern void hostPreBasicFluidKernel( double **dStores, int **nStores, int nPtrs, ... );
	extern void hostAddSource( double *buf1, double *buf2, sge::SGCUDAGRID *grids );
	extern void hostDiffusion( double *buf_old, double *buf_new, sge::SGCUDAGRID *grids,
		const sge::SGGRIDTYPE type, const double diffusion );
	extern void hostAdvection( double *buf, double *dStores, 
		const sge::SGCUDAGRID *grids, const sge::SGGRIDTYPE type );
	extern void hostProject ( double *vel_u, double *vel_v, double *vel_w,
		double *div, double *p, sge::SGCUDAGRID *grids );
#ifdef __cplusplus
}
#endif

#endif