/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Jan 31, 2014
* <File>        FunctionHelper.cu
*/

#ifndef __function_helper_cu_
#define __function_helper_cu_

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

SGRUNTIMEMSG FuncHelper::PrepareBasicFluidKernel( SGINT nPtrs, SGDOUBLE **dStores, ... )
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

#endif



/**
* <Author>      Orlando Chen
* <First>       Nov 21, 2013
* <Last>		Jan 31, 2014
* <File>        InterpKernels.h
*/

#ifndef __interpolation_kernels_h_
#define __interpolation_kernels_h_

#include "CUDADef.h"
#include "CUDAMathLib.h"
#include <cuda_runtime_api.h>

using namespace sge;

__device__	
	double atomicGetValue(
	const SGSTDGRID *buff, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	if ( x < gst_header or x > gst_tailer ) return 0.f;
	if ( y < gst_header or y > gst_tailer ) return 0.f;
	if ( z < gst_header or z > gst_tailer ) return 0.f;

	switch (type)
	{
	case SG_DENSITY_FIELD:
		return buff[ Index(x,y,z) ].dens;
	case SG_VELOCITY_U_FIELD:
		return buff[ Index(x,y,z) ].u;
	case SG_VELOCITY_V_FIELD:
		return buff[ Index(x,y,z) ].v;
	case SG_VELOCITY_W_FIELD:
		return buff[ Index(x,y,z) ].w;
	}
};

__device__
	void atomicSetValue( 
	SGSTDGRID *buff, const double value, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	if ( x < gst_header or x > gst_tailer ) return ;
	if ( y < gst_header or y > gst_tailer ) return ;
	if ( z < gst_header or z > gst_tailer ) return ;

	switch (type)
	{
	case SG_DENSITY_FIELD:
		buff[ Index(x,y,z) ].dens = value;
		break;
	case SG_VELOCITY_U_FIELD:
		buff[ Index(x,y,z) ].u = value;
		break;
	case SG_VELOCITY_V_FIELD:
		buff[ Index(x,y,z) ].v = value;
		break;
	case SG_VELOCITY_W_FIELD:
		buff[ Index(x,y,z) ].w = value;
		break;
	}
};

__device__
	SGUSINGNODE atomicUsingNode( const int x,const int y, const int z )
{
	/* if position at center grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= 0 and y < GRIDS_X and
		z >= 0 and z < GRIDS_X )
		return SG_USING_CENTER;

	/* if position at left grids */
	if ( x >= -GRIDS_X and x < 0 and
		y >= 0 and y < GRIDS_X  and
		z >= 0 and z < GRIDS_X )
		return SG_USING_LEFT;

	/* if position at right grids */
	if ( x >= GRIDS_X and x < GRIDS_X * 2 and
		y >= 0 and y < GRIDS_X  and
		z >= 0 and z < GRIDS_X )
		return SG_USING_RIGHT;

	/* if position at up grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= GRIDS_X and y < GRIDS_X * 2 and
		z >= 0 and z < GRIDS_X )
		return SG_USING_UP;

	/* if position at down grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= -GRIDS_X and y < 0 and
		z >= 0 and z < GRIDS_X )
		return SG_USING_DOWN;

	/* if position at front grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= 0 and y < GRIDS_X and
		z >= GRIDS_X and z < GRIDS_X * 2 )
		return SG_USING_FRONT;

	/* if position at back grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= 0 and y < GRIDS_X and
		z >= -GRIDS_X and z < 0 )
		return SG_USING_BACK;

	return SG_NO_DEFINE;
};

__device__
double atomicGetDeviceBuffer( const SGDEVBUFF *buff, const SGGRIDTYPE type,
	const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* check the bounds */
	if ( x < lower or x >= upper ) return 0.f;
	if ( y < lower or y >= upper ) return 0.f;
	if ( z < lower or z >= upper ) return 0.f;

	/* check the region */
	SGNODECODE coord = atomicCheckNodeCord( x, y, z );
	double value = 0.f;
	
	switch (coord)
	{
	case sge::SG_USING_CENTER:
		if ( buff->ptrCenterGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrCenterGrids, type, x, y, z );
		break;
	case sge::SG_USING_LEFT:
		if ( buff->ptrLeftGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrLeftGrids, type, x + GRIDS_X, y, z );
		break;
	case sge::SG_USING_RIGHT:
		if ( buff->ptrRightGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrRightGrids, type, x - GRIDS_X, y, z );
		break;
	case sge::SG_USING_UP:
		if ( buff->ptrUpGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrUpGrids, type, x, y - GRIDS_X, z );
		break;
	case sge::SG_USING_DOWN:
		if ( buff->ptrDownGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrDownGrids, type, x, y + GRIDS_X, z );
		break;
	case sge::SG_USING_FRONT:
		if ( buff->ptrFrontGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrDownGrids, type, x, y, z - GRIDS_X );
		break;
	case sge::SG_USING_BACK:
		if ( buff->ptrBackGrids not_eq NULL )
			value = atomicGetValueFromGrids( buff->ptrDownGrids, type, x, y, z + GRIDS_X );
		break;
	default:
		break;
	}

	return value;
};

__device__
void atomicSetDeviceBuffer( SGDEVBUFF *buff, const double value, const SGGRIDTYPE type,
	const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* check the bounds */
	if ( x < lower or x >= upper ) return ;
	if ( y < lower or y >= upper ) return ;
	if ( z < lower or z >= upper ) return ;

	/* check the region */
	if ( atomicCheckNodeCord( x, y, z ) eqt SG_USING_CENTER )
	{
		if ( buff->ptrCenterGrids not_eq NULL )
			atomicSetValueToGrids( buff->ptrCenterGrids, value, type, x, y, z );
	}
};


#include "Parameters.h"
#include "CUDAMath.h"

__device__
void atomicPickVertices( double *dStores, const SGDEVBUFF *buff, const SGGRIDTYPE type,
	double const x, double const y, double const z )
{
	int i = sground( x );
	int j = sground( y );
	int k = sground( z );

	v000 = atomicGetDeviceBuffer( buff, type, i, j, k );
	v001 = atomicGetDeviceBuffer( buff, type, i, j+1, k );
	v011 = atomicGetDeviceBuffer( buff, type, i, j+1, k+1 );
	v010 = atomicGetDeviceBuffer( buff, type, i, j, k+1 );

	v100 = atomicGetDeviceBuffer( buff, type, i+1, j, k );
	v101 = atomicGetDeviceBuffer( buff, type, i+1, j+1, k ); 
	v111 = atomicGetDeviceBuffer( buff, type, i+1, j+1, k+1 );
	v110 = atomicGetDeviceBuffer( buff, type, i+1, j, k+1 );
};

__device__
void atomicPickVertices( double *dStores, const SGCUDAGRID *buff, const SGGRIDTYPE type,
	double const x, double const y, double const z )
{
	int i = sground( x );
	int j = sground( y );
	int k = sground( z );

	v000 = atomicGetValueFromGrids( buff, type, i, j, k );
	v001 = atomicGetValueFromGrids( buff, type, i, j+1, k );
	v011 = atomicGetValueFromGrids( buff, type, i, j+1, k+1 );
	v010 = atomicGetValueFromGrids( buff, type, i, j, k+1 );

	v100 = atomicGetValueFromGrids( buff, type, i+1, j, k );
	v101 = atomicGetValueFromGrids( buff, type, i+1, j+1, k ); 
	v111 = atomicGetValueFromGrids( buff, type, i+1, j+1, k+1 );
	v110 = atomicGetValueFromGrids( buff, type, i+1, j, k+1 );
};

__device__
double atomicTrilinear( double *dStores, const SGDEVBUFF *buff, const SGGRIDTYPE type,
	double const x, double const y, double const z )
{
	atomicPickVertices( dStores, buff, type, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = v000 * ( 1 - dx ) + v001 * dx;
	double c10 = v010 * ( 1 - dx ) + v011 * dx;
	double c01 = v100 * ( 1 - dx ) + v101 * dx;
	double c11 = v110 * ( 1 - dx ) + v111 * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

__device__
double atomicTrilinear( double *dStores, const SGCUDAGRID *buff, const SGGRIDTYPE type,
	double const x, double const y, double const z )
{
	atomicPickVertices( dStores, buff, type, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = v000 * ( 1 - dx ) + v001 * dx;
	double c10 = v010 * ( 1 - dx ) + v011 * dx;
	double c01 = v100 * ( 1 - dx ) + v101 * dx;
	double c11 = v110 * ( 1 - dx ) + v111 * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

#endif