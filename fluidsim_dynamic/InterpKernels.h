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
	case sge::SG_DENS_GRID:
		value = buff[ Index(x,y,z) ].den;
		break;
	case sge::SG_VELU_GRID:
		value = buff[ Index(x,y,z) ].u;
		break;
	case sge::SG_VELV_GRID:
		value = buff[ Index(x,y,z) ].v;
		break;
	case sge::SG_VELW_GRID:
		value = buff[ Index(x,y,z) ].w;
		break;
	case sge::SG_DIV_GRID:
		value = buff[ Index(x,y,z) ].div;
		break;
	case sge::SG_PRES_GRID:
		value = buff[ Index(x,y,z) ].p;
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
	case sge::SG_DENS_GRID:
		value = buff[ Index(x,y,z) ].den;
		break;
	case sge::SG_VELU_GRID:
		value = buff[ Index(x,y,z) ].u;
		break;
	case sge::SG_VELV_GRID:
		value = buff[ Index(x,y,z) ].v;
		break;
	case sge::SG_VELW_GRID:
		value = buff[ Index(x,y,z) ].w;
		break;

	default:
		value = 0.f;
		break;
	}

	return value;
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
double atomicGetFromDeviceBuffer( const SGDEVBUFF *buff, const SGGRIDTYPE type,
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
		value = atomicGetValueFromGrids( buff->ptrCenterGrids, type, x, y, z );
		break;
	case sge::SG_USING_LEFT:
		value = atomicGetValueFromGrids( buff->ptrLeftGrids, type, x + GRIDS_X, y, z );
		break;
	case sge::SG_USING_RIGHT:
		value = atomicGetValueFromGrids( buff->ptrRightGrids, type, x - GRIDS_X, y, z );
		break;
	case sge::SG_USING_UP:
		value = atomicGetValueFromGrids( buff->ptrUpGrids, type, x, y - GRIDS_X, z );
		break;
	case sge::SG_USING_DOWN:
		value = atomicGetValueFromGrids( buff->ptrDownGrids, type, x, y + GRIDS_X, z );
		break;
	case sge::SG_USING_FRONT:
		value = atomicGetValueFromGrids( buff->ptrDownGrids, type, x, y, z - GRIDS_X );
		break;
	case sge::SG_USING_BACK:
		value = atomicGetValueFromGrids( buff->ptrDownGrids, type, x, y, z + GRIDS_X );
		break;
	case sge::SG_NO_DEFINE:
		value = 0.f;
		break;
	default:
		break;
	}

	return value;
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

	v000 = atomicGetFromDeviceBuffer( buff, type, i, j, k );
	v001 = atomicGetFromDeviceBuffer( buff, type, i, j+1, k );
	v011 = atomicGetFromDeviceBuffer( buff, type, i, j+1, k+1 );
	v010 = atomicGetFromDeviceBuffer( buff, type, i, j, k+1 );

	v100 = atomicGetFromDeviceBuffer( buff, type, i+1, j, k );
	v101 = atomicGetFromDeviceBuffer( buff, type, i+1, j+1, k ); 
	v111 = atomicGetFromDeviceBuffer( buff, type, i+1, j+1, k+1 );
	v110 = atomicGetFromDeviceBuffer( buff, type, i+1, j, k+1 );
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