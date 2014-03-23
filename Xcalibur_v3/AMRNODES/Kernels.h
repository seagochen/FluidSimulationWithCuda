/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 04, 2014
* <File Name>     Kernels.h
*/

#ifndef __kernels_h__
#define __kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "CUDACustomeMath.h"
#include "MacroDefinition.h"
#include "FluidSimProc.h"


__global__ void kernelZeroGrids( double *grid )
{
	GetIndex3D();
	grid [ Index(i,j,k) ] = 0.f;
};

__global__ void kernelZeroVolumetric( uchar *visual )
{
	GetIndex3D();

	for ( int ii = 0; ii < HNODES_X; ii++ )
	{
		for ( int jj = 0; jj < HNODES_X; jj++ )
		{
			for ( int kk = 0; kk < HNODES_X; kk++ )
			{
				int di = ii * GRIDS_X + i;
				int dj = jj * GRIDS_X + j;
				int dk = kk * GRIDS_X + k;
				
				/* zero data */
				visual[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;
			}
		}
	}
};

__global__ void kernelZeroShareBuffers( double *bufs )
{
	GetIndex1D();

	bufs[i] = 0.f;
};

__global__ void kernelZeroShareBuffers( int *bufs )
{
	GetIndex1D();

	bufs[i] = 0;
}; 

__global__ void kernelCopyGrids( double *src, cdouble *dst )
{
	GetIndex3D();

	src[Index(i,j,k)] = dst[Index(i,j,k)];
};


__device__  double atomicGetValue( cdouble *grid, cint x, cint y, cint z )
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_tailer ) return 0.f;
	if ( y > gst_tailer ) return 0.f;
	if ( z > gst_tailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

__device__  double atomicTrilinear( cdouble *grid, cdouble x, cdouble y, cdouble z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	double v000 = atomicGetValue ( grid, i, j, k );
	double v001 = atomicGetValue ( grid, i, j+1, k );
	double v011 = atomicGetValue ( grid, i, j+1, k+1 );
	double v010 = atomicGetValue ( grid, i, j, k+1 );
	double v100 = atomicGetValue ( grid, i+1, j, k );
	double v101 = atomicGetValue ( grid, i+1, j+1, k );
	double v111 = atomicGetValue ( grid, i+1, j+1, k+1 );
	double v110 = atomicGetValue ( grid, i+1, j, k+1 );

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


__device__ void atomicDensityObs( double *grids, cdouble *obstacle )
{
	GetIndex3D();
	BeginSimArea();
	/* 当前格点有障碍物，且密度大于0 */
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE and grids[Index(i,j,k)] > 0.f )
	{
		int cells  = 0;
		double val = 0; 

		if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) cells++;

		if ( cells > 0 ) val = grids[Index(i,j,k)] / cells;
		else val = 0.f;

		if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i-1,j,k)] += val;
		if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i+1,j,k)] += val;
		if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j-1,k)] += val;
		if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j+1,k)] += val;
		if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j,k-1)] += val;
		if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j,k+1)] += val;

		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__device__ void atomicVelocityObs_U( double *grids, cdouble *obstacle )
{
	GetIndex3D();
	BeginSimArea();
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index(i,j,k)] > 0.f )
		{
			if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i-1,j,k)] = grids[Index(i-1,j,k)] -  grids[Index(i,j,k)];
		}
		else
		{
			if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i+1,j,k)] = grids[Index(i+1,j,k)] -  grids[Index(i,j,k)];
		}
		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__device__ void atomicVelocityObs_V( double *grids, cdouble *obstacle )
{
	GetIndex3D();
	BeginSimArea();
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index(i,j,k)] > 0.f )
		{
			if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j-1,k)] = grids[Index(i,j-1,k)] - grids[Index(i,j,k)];
		}
		else
		{
			if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j+1,k)] = grids[Index(i,j+1,k)] - grids[Index(i,j,k)];
		}
		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__device__ void atomicVelocityObs_W( double *grids, cdouble *obstacle )
{
	GetIndex3D();
	BeginSimArea();
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index(i,j,k)] > 0.f )
		{
			if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j,k-1)] = grids[Index(i,j,k-1)] - grids[Index(i,j,k)];
		}
		else
		{
			if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j,k+1)] = grids[Index(i,j,k+1)] - grids[Index(i,j,k)];
		}
		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__global__ void kernelObstacle( double *grids, cdouble *obstacle, cint field )
{
	switch( field )
	{
	case MACRO_DENSITY:
		atomicDensityObs( grids, obstacle );
		break;

	case MACRO_VELOCITY_U:
		atomicVelocityObs_U( grids, obstacle );
		break;

	case MACRO_VELOCITY_V:
		atomicVelocityObs_V( grids, obstacle );
		break;

	case MACRO_VELOCITY_W:
		atomicVelocityObs_W( grids, obstacle );
		break;

	default:
		break;
	}
};

#include "TestModule.h"

//__global__ void kernelAddSource( double *density, double *vel_u, double *vel_v, double *vel_w )
//{
//	GetIndex3D();
//	BeginSimArea();
//	
//	cint half = GRIDS_X / 2;
//
//	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
//	{
//		/* add source to grids */
//		density[Index(i,j,k)] = DENSITY;
//
//		/* add velocity to grids */
//		if ( i < half )
//			vel_u[Index(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
//		elif( i >= half )
//			vel_u[Index(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;
//
//		vel_v[Index(i,j,k)] = VELOCITY;
//
//		if ( k < half )
//			vel_w[Index(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
//		elif ( k >= half )
//			vel_w[Index(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;
//	}
//	EndSimArea();
//};

__global__ void kernelPickData( uchar *c, cdouble *bufs, int ofi, int ofj, int ofk, cint grids )
{
	GetIndex3D();

	ofi = ofi * grids + i;
	ofj = ofj * grids + j;
	ofk = ofk * grids + k;

	/* zero c first */
	c[cudaIndex3D(ofi,ofj,ofk,VOLUME_X)] = 0;

	/* append c to volume c */
	int temp = atomicRound( bufs[ Index(i, j, k) ] );
	if ( temp > 0 and temp < 250 )
		c [ cudaIndex3D(ofi,ofj,ofk,VOLUME_X) ] = (uchar) temp;
};

__global__ void kernelInterRootGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	GetIndex3D();

	double x = ( pi * GRIDS_X + i ) * rate;
	double y = ( pj * GRIDS_X + j ) * rate;
	double z = ( pk * GRIDS_X + k ) * rate;

	dst[Index(i,j,k)] = atomicTrilinear( src, x, y, z );
};

__global__ void kernelInterLeafGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	GetIndex3D();

	int x = atomicRound( ( pi * GRIDS_X + i ) * rate );
	int y = atomicRound( ( pj * GRIDS_X + j ) * rate );
	int z = atomicRound( ( pk * GRIDS_X + k ) * rate );

	dst[Index(x,y,z)] = src[Index(i,j,k)];
};

__global__ void kernelClearHalo( double *grids )
{
	GetIndex3D();

	grids[Index(gst_header,j,k)] = 0.f;
	grids[Index(gst_tailer,j,k)] = 0.f;
	grids[Index(i,gst_header,k)] = 0.f;
	grids[Index(i,gst_tailer,k)] = 0.f;
	grids[Index(i,j,gst_header)] = 0.f;
	grids[Index(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelHandleHalo
	( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back )
{
	GetIndex3D();

	center[Index(gst_header,j,k)] = left[Index(gst_tailer,j,k)];
	center[Index(gst_tailer,j,k)] = right[Index(gst_header,j,k)];
	center[Index(i,gst_tailer,k)] = up[Index(i,gst_header,k)];
    center[Index(i,gst_header,k)] = down[Index(i,gst_tailer,k)];
	center[Index(i,j,gst_tailer)] = front[Index(i,j,gst_header)];
	center[Index(i,j,gst_header)] = back[Index(i,j,gst_tailer)];

/*	c[cudaIndex3D(gst_header,gst_header,k,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_header,k,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_header,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_header,gst_tailer,k,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_tailer,k,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_tailer,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,gst_header,k,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_header,k,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_header,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,gst_tailer,k,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_tailer,k,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_tailer,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_header,j,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_header,j,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,j,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_header,j,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_header,j,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,j,sim_tailer,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,j,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,j,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,j,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,j,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,j,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,j,sim_tailer,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_header,gst_header,VOLUME_X)] = ( c[cudaIndex3D(i,sim_header,gst_header,VOLUME_X)] + c[cudaIndex3D(i,gst_header,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_header,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(i,sim_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(i,gst_header,sim_tailer,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_tailer,gst_header,VOLUME_X)] = ( c[cudaIndex3D(i,sim_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(i,gst_tailer,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_tailer,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(i,sim_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(i,gst_tailer,sim_tailer,VOLUME_X)] ) / 2.f;

	c[cudaIndex3D(gst_header,gst_header,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_header,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_header,gst_header,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_header,sim_tailer,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_header,gst_tailer,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_tailer,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_header,gst_tailer,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_tailer,sim_tailer,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_header,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_header,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_header,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_header,sim_tailer,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_tailer,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_tailer,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_tailer,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_tailer,sim_tailer,VOLUME_X)] ) / 3.f;
*/
};


__global__ void kernelSumDensity( double *share, cdouble *src, cint no )
{
	GetIndex3D();

	share[no] += src[Index(i,j,k)];
};

#endif