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
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "TestModule.h"

#define IX(i,j,k) ix(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X)
__global__ void kernelZeroGrids( double *grid )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	grid [ IX(i,j,k) ] = 0.f;
};

__global__ void kernelZeroVolumetric( uchar *visual )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

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
				visual[ ix(di, dj, dk, VOLUME_X,VOLUME_X,VOLUME_X) ] = 0;
			}
		}
	}
};

__global__ void kernelZeroShareBuffers( double *bufs )
{
	int i;
	_thread(&i);

	bufs[i] = 0.f;
};

__global__ void kernelZeroShareBuffers( int *bufs )
{
	int i;
	_thread(&i);

	bufs[i] = 0;
}; 

__global__ void kernelCopyGrids( double *src, cdouble *dst )
{
	int i, j, k; 
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	src[IX(i,j,k)] = dst[IX(i,j,k)];
};

__device__ void atomicDensityObs( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	/* 当前格点有障碍物，且密度大于0 */
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE and grids[IX(i,j,k)] > 0.f )
	{
		int cells  = 0;
		double val = 0; 

		if ( obstacle[IX(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) cells++;

		if ( cells > 0 ) val = grids[IX(i,j,k)] / cells;
		else val = 0.f;

		if ( obstacle[IX(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i-1,j,k)] += val;
		if ( obstacle[IX(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i+1,j,k)] += val;
		if ( obstacle[IX(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j-1,k)] += val;
		if ( obstacle[IX(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j+1,k)] += val;
		if ( obstacle[IX(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j,k-1)] += val;
		if ( obstacle[IX(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j,k+1)] += val;

		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__device__ void atomicVelocityObs_U( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[IX(i,j,k)] > 0.f )
		{
			if ( obstacle[IX(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i-1,j,k)] = grids[IX(i-1,j,k)] -  grids[IX(i,j,k)];
		}
		else
		{
			if ( obstacle[IX(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i+1,j,k)] = grids[IX(i+1,j,k)] -  grids[IX(i,j,k)];
		}
		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__device__ void atomicVelocityObs_V( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[IX(i,j,k)] > 0.f )
		{
			if ( obstacle[IX(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j-1,k)] = grids[IX(i,j-1,k)] - grids[IX(i,j,k)];
		}
		else
		{
			if ( obstacle[IX(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j+1,k)] = grids[IX(i,j+1,k)] - grids[IX(i,j,k)];
		}
		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__device__ void atomicVelocityObs_W( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[IX(i,j,k)] > 0.f )
		{
			if ( obstacle[IX(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j,k-1)] = grids[IX(i,j,k-1)] - grids[IX(i,j,k)];
		}
		else
		{
			if ( obstacle[IX(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j,k+1)] = grids[IX(i,j,k+1)] - grids[IX(i,j,k)];
		}
		grids[IX(i,j,k)] = 0.f;
	}
	}
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

__global__ void kernelAddSource( double *density, double *vel_u, double *vel_v, double *vel_w )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	
	cint half = GRIDS_X / 2;

	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
	{
		/* add source to grids */
		density[IX(i,j,k)] = DENSITY;

		/* add velocity to grids */
		if ( i < half )
			vel_u[IX(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
		elif( i >= half )
			vel_u[IX(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;

		vel_v[IX(i,j,k)] = VELOCITY;

		if ( k < half )
			vel_w[IX(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
		elif ( k >= half )
			vel_w[IX(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;
	}
	}
};

__global__ void kernelPickData( uchar *c, cdouble *bufs, int ofi, int ofj, int ofk, cint grids )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	ofi = ofi * grids + i;
	ofj = ofj * grids + j;
	ofk = ofk * grids + k;

	/* zero c first */
	c[ ix(ofi,ofj,ofk,VOLUME_X,VOLUME_X,VOLUME_X) ] = 0;

	/* append c to volume c */
	int temp = _round( bufs[ IX(i, j, k) ] );
	if ( temp > 0 and temp < 250 )
		c [ ix(ofi,ofj,ofk,VOLUME_X,VOLUME_X,VOLUME_X) ] = (uchar) temp;
};

__global__ void kernelInterRootGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	double x = ( pi * GRIDS_X + i ) * rate;
	double y = ( pj * GRIDS_X + j ) * rate;
	double z = ( pk * GRIDS_X + k ) * rate;

	dst[IX(i,j,k)] = atomicTrilinear( src, x, y, z, GRIDS_X, GRIDS_X, GRIDS_X );
};

__global__ void kernelInterLeafGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	int x = _round( ( pi * GRIDS_X + i ) * rate );
	int y = _round( ( pj * GRIDS_X + j ) * rate );
	int z = _round( ( pk * GRIDS_X + k ) * rate );

	dst[IX(x,y,z)] = src[IX(i,j,k)];
};

__global__ void kernelClearHalo( double *grids )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	grids[IX(gst_header,j,k)] = 0.f;
	grids[IX(gst_tailer,j,k)] = 0.f;
	grids[IX(i,gst_header,k)] = 0.f;
	grids[IX(i,gst_tailer,k)] = 0.f;
	grids[IX(i,j,gst_header)] = 0.f;
	grids[IX(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelHandleHalo
	( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	center[IX(gst_header,j,k)] = left[IX(gst_tailer,j,k)];
	center[IX(gst_tailer,j,k)] = right[IX(gst_header,j,k)];
	center[IX(i,gst_tailer,k)] = up[IX(i,gst_header,k)];
    center[IX(i,gst_header,k)] = down[IX(i,gst_tailer,k)];
	center[IX(i,j,gst_tailer)] = front[IX(i,j,gst_header)];
	center[IX(i,j,gst_header)] = back[IX(i,j,gst_tailer)];

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
	int i, j, k; 
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	share[no] += src[IX(i,j,k)];
};

#endif