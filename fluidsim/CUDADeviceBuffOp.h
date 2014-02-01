/**
* <Author>      Orlando Chen
* <First>       Nov 21, 2013
* <Last>		Feb 01, 2014
* <File>        CUDADeviceBuffOp.h
*/

#ifndef __cuda_device_buffer_operations_h_
#define __cuda_device_buffer_operations_h_

#include <cuda_runtime_api.h>
#include "CUDADef.h"
#include "CUDAFunctionHelper.h"

using namespace sge;

__device__ double atomicGetValue( const SGSTDGRID *buff, const SGFIELDTYPE type,
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
//	case SG_DIVERGENCE_FIELD:
//		return buff[ Index(x,y,z) ].div;
//	case SG_PRESSURE_FIELD:
//		return buff[ Index(x,y,z) ].p;
	}
};

__device__ void atomicSetValue( SGSTDGRID *buff, const double value, const SGFIELDTYPE type,
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
//	case SG_DIVERGENCE_FIELD:
//		buff[ Index(x,y,z) ].div = value;
//		break;
//	case SG_PRESSURE_FIELD:
//		buff[ Index(x,y,z) ].p = value;
//		break;
	}
};

__device__ SGNODECOORD atomicNodeCoord( const int x,const int y, const int z )
{
	/* if position at center grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= 0 and y < GRIDS_X and
		z >= 0 and z < GRIDS_X )
		return SG_CENTER;

	/* if position at left grids */
	if ( x >= -GRIDS_X and x < 0 and
		y >= 0 and y < GRIDS_X  and
		z >= 0 and z < GRIDS_X )
		return SG_LEFT;

	/* if position at right grids */
	if ( x >= GRIDS_X and x < GRIDS_X * 2 and
		y >= 0 and y < GRIDS_X  and
		z >= 0 and z < GRIDS_X )
		return SG_RIGHT;

	/* if position at up grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= GRIDS_X and y < GRIDS_X * 2 and
		z >= 0 and z < GRIDS_X )
		return SG_UP;

	/* if position at down grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= -GRIDS_X and y < 0 and
		z >= 0 and z < GRIDS_X )
		return SG_DOWN;

	/* if position at front grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= 0 and y < GRIDS_X and
		z >= GRIDS_X and z < GRIDS_X * 2 )
		return SG_FRONT;

	/* if position at back grids */
	if ( x >= 0 and x < GRIDS_X and
		y >= 0 and y < GRIDS_X and
		z >= -GRIDS_X and z < 0 )
		return SG_BACK;

	return SG_NO_DEFINE;
};

__device__ double atomicGetDeviceBuffer( const SGDEVICEBUFF *buff, const SGFIELDTYPE type,
										const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* check the bounds */
	if ( x < lower or x >= upper ) return 0.f;
	if ( y < lower or y >= upper ) return 0.f;
	if ( z < lower or z >= upper ) return 0.f;

	/* check the region */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );
	double value = 0.f;
	
	switch (coord)
	{
	case SG_CENTER:
		if ( buff->ptrCenter not_eq NULL )
			value = atomicGetValue( buff->ptrCenter, type, x, y, z );
		break;
	case SG_LEFT:
		if ( buff->ptrLeft not_eq NULL )
			value = atomicGetValue( buff->ptrLeft, type, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT:
		if ( buff->ptrRight not_eq NULL )
			value = atomicGetValue( buff->ptrRight, type, x - GRIDS_X, y, z );
		break;
	case SG_UP:
		if ( buff->ptrUp not_eq NULL )
			value = atomicGetValue( buff->ptrUp, type, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:
		if ( buff->ptrDown not_eq NULL )
			value = atomicGetValue( buff->ptrDown, type, x, y + GRIDS_X, z );
		break;
	case SG_FRONT:
		if ( buff->ptrFront not_eq NULL )
			value = atomicGetValue( buff->ptrFront, type, x, y, z - GRIDS_X );
		break;
	case sge::SG_BACK:
		if ( buff->ptrBack not_eq NULL )
			value = atomicGetValue( buff->ptrBack, type, x, y, z + GRIDS_X );
		break;
	default:
		break;
	}

	return value;
};

__device__ void atomicSetDeviceBuffer( SGDEVICEBUFF *buff, const double value, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* check the bounds */
	if ( x < lower or x >= upper ) return ;
	if ( y < lower or y >= upper ) return ;
	if ( z < lower or z >= upper ) return ;

	/* check the region */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );

	switch (coord)
	{
	case SG_CENTER:
		if ( buff->ptrCenter not_eq NULL )
			atomicSetValue( buff->ptrCenter, value, type, x, y, z );
		break;
	case SG_LEFT:
		if ( buff->ptrLeft not_eq NULL )
			atomicSetValue( buff->ptrLeft, value, type, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT:
		if ( buff->ptrRight not_eq NULL )
			atomicSetValue( buff->ptrRight, value, type, x - GRIDS_X, y, z );
		break;
	case SG_UP:
		if ( buff->ptrUp not_eq NULL )
			atomicSetValue( buff->ptrUp, value, type, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:
		if ( buff->ptrDown not_eq NULL )
			atomicSetValue( buff->ptrDown, value, type, x, y + GRIDS_X, z );
		break;
	case SG_FRONT:
		if ( buff->ptrFront not_eq NULL )
			atomicSetValue( buff->ptrFront, value, type, x, y, z - GRIDS_X );
		break;
	case sge::SG_BACK:
		if ( buff->ptrBack not_eq NULL )
			atomicSetValue( buff->ptrBack, value, type, x, y, z + GRIDS_X );
		break;
	default:
		break;
	}
};

#include "CUDAMathLib.h"

#define v000  dStores[ 0 ]
#define v001  dStores[ 1 ]
#define v011  dStores[ 2 ]
#define v010  dStores[ 3 ]
#define v100  dStores[ 4 ]
#define v101  dStores[ 5 ]
#define v111  dStores[ 6 ]
#define v110  dStores[ 7 ]

__device__ void atomicPickVertices( double *dStores, const SGDEVICEBUFF *buff, const SGFIELDTYPE type,
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
double atomicTrilinear( double *dStores, const SGDEVICEBUFF *buff, const SGFIELDTYPE type,
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

#undef v000
#undef v001
#undef v011
#undef v010
#undef v100
#undef v101
#undef v111
#undef v110

__global__ void kernelCopyBuffer( double *buff, const SGSTDGRID *grids, const SGFIELDTYPE type )
{
	GetIndex();

	buff[ Index(i,j,k) ] = atomicGetValue( grids, type, i, j, k );
};

__global__ void kernelCopyBuffer( SGSTDGRID *grids, const double *buff, const SGFIELDTYPE type )
{
	GetIndex();

	double value = buff[ Index(i,j,k) ];
	atomicSetValue( grids, value, type, i, j, k );
};

__global__ void kernelSwapBuffer ( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};


#endif