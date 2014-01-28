/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Jan 26, 2014
* <File>        FunctionHelper.cpp
*/

#include <cuda_runtime_api.h>
#include "FunctionHelper.h"

using namespace sge;

void CUDAHelper::CheckErrors( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};

void CUDAHelper::DeviceDim2D( dim3 *grid_out, dim3 *block_out )
{
	block_out->x = TILE_X;
	block_out->y = TILE_X;
	grid_out->x  = GRIDS_X / TILE_X;
	grid_out->y  = GRIDS_X / TILE_X;
};

void CUDAHelper::DeviceDim3D( dim3 *gridDim, dim3 *blockDim )
{
	blockDim->x = (GRIDS_X / TILE_X);
	blockDim->y = (THREADS_X / TILE_X);
	gridDim->x  = (GRIDS_X / blockDim->x);
	gridDim->y  = (GRIDS_X * GRIDS_X * GRIDS_X) / 
		(blockDim->x * blockDim->y * (GRIDS_X / blockDim->x));
};

#include <stdarg.h>
#include <memory>
#include <SGE\config\ISO646.h>

string AppHelper::string_fmt( const std::string fmt_str, ... )
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