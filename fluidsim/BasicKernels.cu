/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 01, 2014
* <File Name>     BasicKernels.cu
*/

#ifndef __basic_kernels_cu_
#define __basic_kernels_cu_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "DataStructures.h"
#include "CUDAMacroDef.h"


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
		value = 0.f;
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


/**
* Do trilinear interpolation at global
*
*/


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

__global__ void kernelPickData ( unsigned char *data, double const *grid, 
	int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};


using namespace sge;

CUDAFuncHelper m_helper;

__global__ void kernelAddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	GetIndex();

	if ( grids[Index(i,j,k)].obstacle eqt SG_BD_SOURCE )
	{
		switch ( type )
		{
		case SG_DENSITY_FIELD:
			buffer[Index(i,j,k)] = SOURCE_DENSITY;
			break;
		case SG_VELOCITY_V_FIELD:
			buffer[Index(i,j,k)] = SOURCE_VELOCITY;
			break;

		default:
			break;
		}
	}
}

__device__ void atomicDensity( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k  )
{
	int ix = 0;
	if ( grids[Index(i+1,j,k)].obstacle not_eq SG_BD_WALL ) ix++;
	if ( grids[Index(i-1,j,k)].obstacle not_eq SG_BD_WALL ) ix++;
	if ( grids[Index(i,j+1,k)].obstacle not_eq SG_BD_WALL ) ix++;
	if ( grids[Index(i,j-1,k)].obstacle not_eq SG_BD_WALL ) ix++;
	if ( grids[Index(i,j,k+1)].obstacle not_eq SG_BD_WALL ) ix++;
	if ( grids[Index(i,j,k-1)].obstacle not_eq SG_BD_WALL ) ix++;

	if ( ix eqt 0 )
	{
		buffer[Index(i,j,k)] = 0.f;
		return;
	}

	if ( grids[Index(i+1,j,k)].obstacle not_eq SG_BD_WALL ) buffer[Index(i+1,j,k)] += buffer[Index(i,j,k)] / ix;
	if ( grids[Index(i-1,j,k)].obstacle not_eq SG_BD_WALL ) buffer[Index(i-1,j,k)] += buffer[Index(i,j,k)] / ix;
	if ( grids[Index(i,j+1,k)].obstacle not_eq SG_BD_WALL ) buffer[Index(i,j+1,k)] += buffer[Index(i,j,k)] / ix;
	if ( grids[Index(i,j-1,k)].obstacle not_eq SG_BD_WALL ) buffer[Index(i,j-1,k)] += buffer[Index(i,j,k)] / ix;
	if ( grids[Index(i,j,k+1)].obstacle not_eq SG_BD_WALL ) buffer[Index(i,j,k+1)] += buffer[Index(i,j,k)] / ix;
	if ( grids[Index(i,j,k-1)].obstacle not_eq SG_BD_WALL ) buffer[Index(i,j,k-1)] += buffer[Index(i,j,k)] / ix;

	buffer[Index(i,j,k)] = 0.f;
};

__device__ void atomicVelocity_U( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	if ( buffer[Index(i,j,k)] >= 0.f )
	{
		if ( grids[Index(i-1,j,k)].obstacle not_eq SG_BD_WALL )
		{
			buffer[Index(i-1,j,k)] += -buffer[Index(i,j,k)];
		}
	}
	else
	{
		if ( grids[Index(i+1,j,k)].obstacle not_eq SG_BD_WALL )
		{
			buffer[Index(i+1,j,k)] += -buffer[Index(i,j,k)];
		}
	}
};

__device__ void atomicVelocity_V( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	if ( buffer[Index(i,j,k)] >= 0.f )
	{
		if ( grids[Index(i,j-1,k)].obstacle not_eq SG_BD_WALL )
		{
			buffer[Index(i,j-1,k)] += -buffer[Index(i,j,k)];
		}
	}
	else
	{
		if ( grids[Index(i,j+1,k)].obstacle not_eq SG_BD_WALL )
		{
			buffer[Index(i,j+1,k)] += -buffer[Index(i,j,k)];
		}
	}

	buffer[Index(i,j,k)] = 0.f;
};

__device__ void atomicVelocity_W( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	if ( buffer[Index(i,j,k)] >= 0.f )
	{
		if ( grids[Index(i,j,k-1)].obstacle not_eq SG_BD_WALL )
		{
			buffer[Index(i,j,k-1)] += -buffer[Index(i,j,k)];
		}
	}
	else
	{
		if ( grids[Index(i,j,k+1)].obstacle not_eq SG_BD_WALL )
		{
			buffer[Index(i,j,k+1)] += -buffer[Index(i,j,k)];
		}
	}

	buffer[Index(i,j,k)] = 0.f;
};

__global__ void kernelBoundary( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	GetIndex();

	if ( grids[Index(i,j,k)].obstacle eqt SG_BD_WALL )
	{
		switch ( type )
		{
		case SG_DENSITY_FIELD:
			atomicDensity( buffer, grids, i, j, k );
			break;
		case SG_VELOCITY_U_FIELD:
			atomicVelocity_U( buffer, grids, i, j, k );
			break;
		case SG_VELOCITY_V_FIELD:
			atomicVelocity_V( buffer, grids, i, j, k );
			break;
		case SG_VELOCITY_W_FIELD:
			atomicVelocity_W( buffer, grids, i, j, k );
			break;
		default:
			break;
		}
	}
};

__global__ void kernelSmoothHalo( double *buffer, double *stores, SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	GetIndex();

	buffer[ Index(i,j,gst_header) ] += atomicTrilinear( stores, global, type, i, j, gst_header );
	buffer[ Index(i,j,gst_tailer) ] += atomicTrilinear( stores, global, type, i, j, gst_tailer );
	buffer[ Index(i,gst_header,k) ] += atomicTrilinear( stores, global, type, i, gst_header, k );
	buffer[ Index(i,gst_tailer,k) ] += atomicTrilinear( stores, global, type, i, gst_tailer, k );
	buffer[ Index(gst_header,j,k) ] += atomicTrilinear( stores, global, type, gst_header, j, k );
	buffer[ Index(gst_tailer,j,k) ] += atomicTrilinear( stores, global, type, gst_tailer, j, k );

	if ( i eqt gst_header or i eqt gst_tailer ) 
	{
		if ( j eqt gst_header or j eqt gst_tailer )
		{
			if ( k eqt gst_header or k eqt gst_tailer )
			{
				/* vertices */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 3.f;
			}
			elif ( k not_eq gst_header and k not_eq gst_tailer )
			{
				/* edges x 4 */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 2.f;
			}
		}
	}

	if ( i eqt gst_header or i eqt gst_tailer ) 
	{
		if ( k eqt gst_header or k eqt gst_tailer )
		{
			if ( j not_eq gst_header and j not_eq gst_tailer )
			{
				/* edges x 4 */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 2.f;
			}
		}
	}

	if ( j eqt gst_header or j eqt gst_tailer )
	{
		if ( k eqt gst_header or k eqt gst_tailer )
		{
			if ( i not_eq gst_header and i not_eq gst_tailer )
			{
				/* edges x 4 */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 2.f;
			}
		}
	}
};

__host__ void hostBoundary( double *buffer, double *stores, SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	
	kernelBoundary<<<gridDim, blockDim>>>( buffer, global->ptrCenter, type );
	kernelSmoothHalo<<<gridDim, blockDim>>>( buffer, stores, global, type );
};

__global__ void kernelJacobi( double *grid_out, double const *grid_in, 
							 double const diffusion, double const divisor )
{
	GetIndex();
	BeginSimArea();

	double div = 0.f;
	if ( divisor <= 0.f ) div = 1.f;
	else div = divisor;

	grid_out [ Index(i,j,k) ] = 
		( grid_in [ Index(i,j,k) ] + diffusion * 
			(
				grid_out [ Index(i-1, j, k) ] + grid_out [ Index(i+1, j, k) ] +
				grid_out [ Index(i, j-1, k) ] + grid_out [ Index(i, j+1, k) ] +
				grid_out [ Index(i, j, k-1) ] + grid_out [ Index(i, j, k+1) ]
			) 
		) / div;

	EndSimArea();
}

__host__ void hostJacobi ( double *grid_out, double const *grid_in, double const diffusion )
{
	double rate = diffusion;

	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	for ( int k=0; k<20; k++)
		kernelJacobi <<<gridDim, blockDim>>> (grid_out, grid_in, rate, 1+6*rate);
};

__global__ void kernelAdvection( double *buffer, double *stores,
								const SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	GetIndex();
	
	double u = i - global->ptrCenter[ Index(i,j,k) ].u * DELTATIME;
	double v = j - global->ptrCenter[ Index(i,j,k) ].v * DELTATIME;
	double w = k - global->ptrCenter[ Index(i,j,k) ].w * DELTATIME;
	buffer[ Index(i,j,k) ] = atomicTrilinear( stores, global, type, u, v, w );
};

__host__ void hostAdvection( double *buffer, double *stores,
								const SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	kernelAdvection <<<gridDim, blockDim>>> ( buffer, stores, global, type );
};

__global__ void kernelGradient( double *div, double *p, 
	double const *vel_u, double const *vel_v, double const *vel_w )
{
	GetIndex();
	BeginSimArea();
	
	const double h = 1.f / GRIDS_X;

	// previous instantaneous magnitude of velocity gradient 
	//		= (sum of velocity gradients per axis)/2N:
	div [ Index(i,j,k) ] = -0.5f * h * (
			vel_u [ Index(i+1, j, k) ] - vel_u [ Index(i-1, j, k) ] + // gradient of u
			vel_v [ Index(i, j+1, k) ] - vel_v [ Index(i, j-1, k) ] + // gradient of v
			vel_w [ Index(i, j, k+1) ] - vel_w [ Index(i, j, k-1) ]   // gradient of w
		);
	// zero out the present velocity gradient
	p [ Index(i,j,k) ] = 0.f;
	
	EndSimArea();
};

__global__ void kernelSubtract( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__ void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );

	// the velocity gradient
	kernelGradient <<<gridDim, blockDim>>> (div, p, vel_u, vel_v, vel_w);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 1.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract <<<gridDim, blockDim>>> (vel_u, vel_v, vel_w, p);
};

__global__ void kernelOutFlow( double *buffer, const SGDEVICEBUFF *global, SGFIELDTYPE type, SGNODECOORD coord )
{
	GetIndex();

	switch (coord)
	{
	case sge::SG_LEFT:
		buffer[Index(gst_tailer, j, k)] += atomicGetDeviceBuffer( global, type, gst_header, j, k );
		buffer[Index(gst_tailer, j, k)] /= 2.f;
		break;
	case sge::SG_RIGHT:
		buffer[Index(gst_header, j, k)] += atomicGetDeviceBuffer( global, type, gst_tailer, j, k );
		buffer[Index(gst_header, j, k)] /= 2.f;
		break;
	case sge::SG_UP:
		buffer[Index(i, gst_header, k)] += atomicGetDeviceBuffer( global, type, i, gst_tailer, k );
		buffer[Index(i, gst_header, k)] /= 2.f;
		break;
	case sge::SG_DOWN:
		buffer[Index(i, gst_tailer, k)] += atomicGetDeviceBuffer( global, type, i, gst_header, k );
		buffer[Index(i, gst_tailer, k)] /= 2.f;
		break;
	case sge::SG_FRONT:
		buffer[Index(i, j, gst_header)] += atomicGetDeviceBuffer( global, type, i, j, gst_tailer );
		buffer[Index(i, j, gst_header)] /= 2.f;
		break;
	case sge::SG_BACK:
		buffer[Index(i, j, gst_tailer)] += atomicGetDeviceBuffer( global, type, i, j, gst_header );
		buffer[Index(i, j, gst_tailer)] /= 2.f;
		break;

	default:
		break;
	}


};


namespace sge
{
	__host__ void VelocitySolver( double *u, double *v, double *w, double *div, double *p,
		double *u0, double *v0, double *w0,
		SGDEVICEBUFF *global, double *stores )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
		
		/* copy data to temporary buffer */
		m_helper.CopyData( u, global, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( v, global, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( w, global, SG_VELOCITY_W_FIELD, SG_CENTER );
		
		/* diffuse the velocity field */
		hostJacobi( u0, u, VISOCITY );
		hostJacobi( v0, v, VISOCITY );
		hostJacobi( w0, w, VISOCITY );
		
		kernelSwapBuffer<<<gridDim, blockDim>>> ( u0, u );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( v0, v );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( w0, w );

		/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
		hostProject( u, v, w, div, p );

		/* boundary condition */
		hostBoundary( u, stores, global, SG_VELOCITY_U_FIELD );
		hostBoundary( v, stores, global, SG_VELOCITY_V_FIELD );
		hostBoundary( w, stores, global, SG_VELOCITY_W_FIELD );
		
		/* retrieve data */
		m_helper.CopyData( global, u, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( global, v, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( global, w, SG_VELOCITY_W_FIELD, SG_CENTER );
		
		/* advect the velocity field (per axis): */
		hostAdvection( u0, stores, global, SG_VELOCITY_U_FIELD );
		hostAdvection( v0, stores, global, SG_VELOCITY_V_FIELD );
		hostAdvection( w0, stores, global, SG_VELOCITY_W_FIELD );

		/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
		hostProject ( u0, v0, w0, div, p );

		kernelSwapBuffer<<<gridDim, blockDim>>> ( u0, u );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( v0, v );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( w0, w );

		/* boundary condition */
		hostBoundary( u, stores, global, SG_VELOCITY_U_FIELD );
		hostBoundary( v, stores, global, SG_VELOCITY_V_FIELD );
		hostBoundary( w, stores, global, SG_VELOCITY_W_FIELD );

		/* retrieve data */
		m_helper.CopyData( global, u, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( global, v, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( global, w, SG_VELOCITY_W_FIELD, SG_CENTER );
	}

	__host__ void DensitySolver( double *dens, double *dens0, SGDEVICEBUFF *global, double *stores )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	
		/* copy data to temporary buffer */
		m_helper.CopyData( dens, global, SG_DENSITY_FIELD, SG_CENTER );
	
		/* advection */
		hostJacobi( dens0, dens, DIFFUSION );
		kernelSwapBuffer <<<gridDim, blockDim>>> ( dens0, dens );
		hostAdvection( dens, stores, global, SG_DENSITY_FIELD );

		/* boundary condition */
		hostBoundary( dens, stores, global, SG_DENSITY_FIELD );

		/* retrive data */
		m_helper.CopyData( global, dens, SG_DENSITY_FIELD, SG_CENTER );
	};

	__host__ void AddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
		kernelAddSource <<<gridDim, blockDim>>> ( buffer, grids, type );
	};


	__host__ void HaloDataExchange( double *buffer, SGDEVICEBUFF *global, SGFIELDTYPE type, SGNODECOORD coord )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );

		m_helper.CopyData( buffer, global, type, coord );
		kernelOutFlow<<<gridDim, blockDim>>>( buffer, global, type, coord );
		m_helper.CopyData( global, buffer, type, coord );
	};

};

#endif