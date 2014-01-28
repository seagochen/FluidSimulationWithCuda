/**
* <Author>      Orlando Chen
* <First>       Nov 21, 2013
* <Last>		Jan 28, 2014
* <File>        InterpKernels.h
*/

#ifndef __interpolation_kernels_h_
#define __interpolation_kernels_h_

#include "CUDADef.h"
#include "DataStructures.h"
#include <cuda_runtime_api.h>

using namespace sge;

__device__	
double atomicGetValueFromGrids( const GRIDCPX *buff, const SGGRIDTYPE type,
	const int x, const int y, const int z )
{
	if ( x < gst_header or x > gst_tailer ) return 0.f;
	if ( y < gst_header or y > gst_tailer ) return 0.f;
	if ( z < gst_header or z > gst_tailer ) return 0.f;

	double value = 0.f;

	switch (type)
	{
	case SG_DENS_GRID:
		value = buff[ Index(x,y,z) ].den;
		break;
	case SG_VELU_GRID:
		value = buff[ Index(x,y,z) ].u;
		break;
	case SG_VELV_GRID:
		value = buff[ Index(x,y,z) ].v;
		break;
	case SG_VELW_GRID:
		value = buff[ Index(x,y,z) ].w;
		break;
	case SG_DIV_GRID:
		value = buff[ Index(x,y,z) ].div;
		break;
	case SG_PRES_GRID:
		value = buff[ Index(x,y,z) ].p;
		break;
	case SG_DENS_GRID0:
		value = buff[ Index(x,y,z) ].den0;
		break;
	case SG_VELU_GRID0:
		value = buff[ Index(x,y,z) ].u0;
		break;
	case SG_VELV_GRID0:
		value = buff[ Index(x,y,z) ].v0;
		break;
	case SG_VELW_GRID0:
		value = buff[ Index(x,y,z) ].w0;
		break;


	default:
		value = 0.f;
		break;
	}

	return value;
};

__device__	
double atomicGetValueFromGrids( const GRIDSPL *buff, const SGGRIDTYPE type,
	const int x, const int y, const int z )
{
	if ( x < gst_header or x > gst_tailer ) return 0.f;
	if ( y < gst_header or y > gst_tailer ) return 0.f;
	if ( z < gst_header or z > gst_tailer ) return 0.f;

	double value = 0.f;

	switch (type)
	{
	case SG_DENS_GRID:
		value = buff[ Index(x,y,z) ].den;
		break;
	case SG_VELU_GRID:
		value = buff[ Index(x,y,z) ].u;
		break;
	case SG_VELV_GRID:
		value = buff[ Index(x,y,z) ].v;
		break;
	case SG_VELW_GRID:
		value = buff[ Index(x,y,z) ].w;
		break;

	default:
		value = 0.f;
		break;
	}

	return value;
};

__device__	
void atomicSetValueToGrids( GRIDCPX *buff, const double value, const SGGRIDTYPE type,
	const int x, const int y, const int z )
{
	if ( x < gst_header or x > gst_tailer ) return ;
	if ( y < gst_header or y > gst_tailer ) return ;
	if ( z < gst_header or z > gst_tailer ) return ;

	switch (type)
	{
	case SG_DENS_GRID:
		buff[ Index(x,y,z) ].den = value;
		break;
	case SG_VELU_GRID:
		buff[ Index(x,y,z) ].u = value;
		break;
	case SG_VELV_GRID:
		buff[ Index(x,y,z) ].v = value;
		break;
	case SG_VELW_GRID:
		buff[ Index(x,y,z) ].w = value;
		break;
	case SG_DIV_GRID:
		buff[ Index(x,y,z) ].div = value;
		break;
	case SG_PRES_GRID:
		buff[ Index(x,y,z) ].p = value;
		break;
	case SG_DENS_GRID0:
		buff[ Index(x,y,z) ].den0 = value;
		break;
	case SG_VELU_GRID0:
		buff[ Index(x,y,z) ].u0 = value;
		break;
	case SG_VELV_GRID0:
		buff[ Index(x,y,z) ].v0 = value;
		break;
	case SG_VELW_GRID0:
		buff[ Index(x,y,z) ].w0 = value;
		break;
	}
};

__device__
SGNODECODE atomicCheckNodeCord( const int x,const int y, const int z )
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

#endif