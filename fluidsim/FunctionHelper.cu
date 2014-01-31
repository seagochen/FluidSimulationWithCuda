/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Jan 31, 2014
* <File>        FunctionHelper.cu
*/

#include "DataStructures.h"
#include "FunctionHelper.h"
#include <stdarg.h>
#include <memory>
#include <SGE\config\ISO646.h>
#include <cuda_runtime_api.h>

using namespace sge;

SGVOID FuncHelper::CheckErrors( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};

SGVOID FuncHelper::DeviceDim2D( dim3 *grid_out, dim3 *block_out )
{
	block_out->x = TILE_X;
	block_out->y = TILE_X;
	grid_out->x  = GRIDS_X / TILE_X;
	grid_out->y  = GRIDS_X / TILE_X;
};

SGVOID FuncHelper::DeviceDim3D( dim3 *gridDim, dim3 *blockDim )
{
	blockDim->x = (GRIDS_X / TILE_X);
	blockDim->y = (THREADS_X / TILE_X);
	gridDim->x  = (GRIDS_X / blockDim->x);
	gridDim->y  = (GRIDS_X * GRIDS_X * GRIDS_X) / 
		(blockDim->x * blockDim->y * (GRIDS_X / blockDim->x));
};

string FuncHelper::string_fmt( const std::string fmt_str, ... )
{
	/* reserve 2 times as much as the length of the fmt_str */
    int final_n, n = fmt_str.size() * 2; 
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while ( true )
	{
		/* wrap the plain char array into the unique_ptr */
        formatted.reset ( new char[n] ); 
        strcpy ( &formatted[0], fmt_str.c_str() );
        va_start ( ap, fmt_str );
        final_n = vsnprintf ( &formatted[0], n, fmt_str.c_str(), ap );
        va_end ( ap );
        if ( final_n < 0 or final_n >= n )
            n += abs( final_n - n + 1 );
        else
            break;
    }
    return std::string ( formatted.get() );
};

SGRUNTIMEMSG FuncHelper::PreBasicFluidKernel( SGINT nPtrs, SGDOUBLE **dStores, ... )
{
	/* create CUDA temporary buffer */
	if ( cudaMalloc( (void**)dStores, sizeof(double) * TPBUFFER_X ) != cudaSuccess )
	{
		CheckErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
		return SG_MALLOC_SPACE_FAILED;
	}

	/* malloc device buffer for fluid simulation */
	double **ptr;
	va_list ap;
	const size_t size = GRIDS_X * GRIDS_X * GRIDS_X;
	va_start( ap, dStores );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		if ( cudaMalloc( (void**)ptr, sizeof(double) * size ) != cudaSuccess )
		{
			CheckErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

#include "CUDAMathLib.h"

