/**
* <Author>      Orlando Chen
* <First>       Jan 26, 2013
* <Last>		Jan 26, 2013
* <File>        BasicFluidKernels.h
*/

#ifndef __basic_fluid_kernels_h_
#define __basic_fluid_kernels_h_

#include "DataStructures.h"
#include "FunctionHelper.h"

namespace sge
{

#ifdef __cplusplus
extern "C"
{
#endif
	extern void hostPreBasicFluidKernel( double **dStores, int **nStores );
	extern void hostAddSource( CUDAGRID *grid );
	extern void hostJacobi( CUDAGRID *grid, double *dStores, 
		int const type, double const diffusion, double const divisor );

#ifdef __cplusplus
}
#endif
};

#endif