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

__device__  void atomicVertices
	( double *c000, double *c001, double *c011, double *c010, double *c100, double *c101, double *c111,
	double *c110, cdouble *grid, cdouble x, cdouble y, cdouble z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	*c000 = atomicGetValue ( grid, i, j, k );
	*c001 = atomicGetValue ( grid, i, j+1, k );
	*c011 = atomicGetValue ( grid, i, j+1, k+1 );
	*c010 = atomicGetValue ( grid, i, j, k+1 );
	*c100 = atomicGetValue ( grid, i+1, j, k );
	*c101 = atomicGetValue ( grid, i+1, j+1, k );
	*c111 = atomicGetValue ( grid, i+1, j+1, k+1 );
	*c110 = atomicGetValue ( grid, i+1, j, k+1 );
}

__device__  double atomicTrilinear( cdouble *grid, cdouble x, cdouble y, cdouble z )
{
	double v000, v001, v010, v011, v100, v101, v110, v111;
	atomicVertices ( &v000, &v001, &v011, &v010,
		&v100, &v101, &v111, &v110,
		grid, x, y, z );

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

__global__ void kernelJacobi( double *grid_out, cdouble *grid_in, cdouble diffusion, cdouble divisor )
{
	GetIndex3D();
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

__global__ void kernelGridAdvection( double *grid_out, cdouble *grid_in, cdouble *u_in, cdouble *v_in, cdouble *w_in )
{
	GetIndex3D();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;
	
	grid_out [ Index(i,j,k) ] = atomicTrilinear ( grid_in, u, v, w );

	EndSimArea();
};

__global__ void kernelGradient( double *div, double *p, cdouble *vel_u, cdouble *vel_v, cdouble *vel_w )
{
	GetIndex3D();
	BeginSimArea();
	
	cdouble h = 1.f / GRIDS_X;

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

__global__ void kernelSubtract( double *vel_u, double *vel_v, double *vel_w, cdouble *p )
{
	GetIndex3D();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__global__ void kernelSetBoundary( double *grids )
{
	GetIndex3D();

	BeginSimArea();
	
	cint half = GRIDS_X / 2;

#if !TESTING_MODE_SWITCH
	
	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
		grids[ Index(i,j,k) ] = MACRO_BOUNDARY_SOURCE;
#else
	if ( i >= half-2 and i < half+2 and
		j >= half-2 and j < half+2 and
		k >= half-2 and k < half+2 )
		grids[Index(i,j,k)] = MACRO_BOUNDARY_SOURCE;
#endif

	EndSimArea();
};

__global__ void kernelAddSource( double *density, double *vel_u, double *vel_v, double *vel_w, double *obs )
{
	GetIndex3D();
	BeginSimArea();

	if ( obs[ Index(i,j,k) ] eqt MACRO_BOUNDARY_SOURCE )
	{
		/* add source to grids */
		density[Index(i,j,k)] = SOURCE_DENSITY;

#if !TESTING_MODE_SWITCH
	cint half = GRIDS_X / 2;


		/* add velocity to grids */
		if ( i < half )
			vel_u[Index(i,j,k)] = -SOURCE_VELOCITY * DELTATIME * DELTATIME;
		elif( i >= half )
			vel_u[Index(i,j,k)] =  SOURCE_VELOCITY * DELTATIME * DELTATIME;

		vel_v[Index(i,j,k)] = SOURCE_VELOCITY;

		if ( k < half )
			vel_w[Index(i,j,k)] = -SOURCE_VELOCITY * DELTATIME * DELTATIME;
		elif ( k >= half )
			vel_w[Index(i,j,k)] =  SOURCE_VELOCITY * DELTATIME * DELTATIME;
#else

	/* velocity: default-up(0) down(1) left(2) right(3) front(4) back(5) */
#if TESTING_MODE==0
		vel_v[Index(i,j,k)] =  SOURCE_VELOCITY;
#elif TESTING_MODE==1
		vel_v[Index(i,j,k)] = -SOURCE_VELOCITY;
#elif TESTING_MODE==2
		vel_u[Index(i,j,k)] = -SOURCE_VELOCITY;
#elif TESTING_MODE==3
		vel_u[Index(i,j,k)] =  SOURCE_VELOCITY;
#elif TESTING_MODE==4
		vel_w[Index(i,j,k)] =  SOURCE_VELOCITY;
#elif TESTING_MODE==5
		vel_w[Index(i,j,k)] = -SOURCE_VELOCITY;
#endif

#endif
	}
	EndSimArea();
};

void hostJacobi( double *grid_out, cdouble *grid_in, cdouble *obstacle, cint field, cdouble diffusion, cdouble divisor )
{
	dim3 gridDim, blockDim;
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi<<<gridDim,blockDim>>>(grid_out, grid_in, diffusion, divisor);
	}
	kernelObstacle<<<gridDim,blockDim>>>( grid_out, obstacle, field );
};

void hostAdvection
	( double *grid_out, cdouble *grid_in, cdouble *obstacle, cint field, 
	cdouble *u_in, cdouble *v_in, cdouble *w_in )
{
	dim3 gridDim, blockDim;
	cudaDeviceDim3D();
	kernelGridAdvection<<<gridDim,blockDim>>>( grid_out, grid_in, u_in, v_in, w_in );
	kernelObstacle<<<gridDim,blockDim>>>( grid_out, obstacle, field );
};

void hostDiffusion( double *grid_out, cdouble *grid_in, cdouble diffusion, cdouble *obstacle, cint field )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, obstacle, field, rate, 1+6*rate );
};

void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p, cdouble *obs )
{
	dim3 gridDim, blockDim;
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient<<<gridDim,blockDim>>>( div, p, vel_u, vel_v, vel_w );
	kernelObstacle<<<gridDim,blockDim>>>( div, obs, MACRO_SIMPLE );
	kernelObstacle<<<gridDim,blockDim>>>( p, obs, MACRO_SIMPLE );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, obs, MACRO_SIMPLE, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract<<<gridDim,blockDim>>>( vel_u, vel_v, vel_w, p );
	kernelObstacle<<<gridDim,blockDim>>>( vel_u, obs, MACRO_VELOCITY_U );
	kernelObstacle<<<gridDim,blockDim>>>( vel_v, obs, MACRO_VELOCITY_V );
	kernelObstacle<<<gridDim,blockDim>>>( vel_w, obs, MACRO_VELOCITY_W );
};

__global__ void kernelZeroGrids( double *grid )
{
	GetIndex3D();
	grid [ Index(i,j,k) ] = 0.f;
};

__global__ void kernelZeroVolumetric( sge::SGUCHAR *visual )
{
	GetIndex3D();

	for ( int ii = 0; ii < NODES_X; ii++ )
	{
		for ( int jj = 0; jj < NODES_X; jj++ )
		{
			for ( int kk = 0; kk < NODES_X; kk++ )
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

__global__ void kernelZeroTemporaryBuffers( double *bufs )
{
	GetIndex1D();

	bufs[i] = 0.f;
};

__global__ void kernelZeroTemporaryBuffers( int *bufs )
{
	GetIndex1D();

	bufs[i] = 0;
};

__global__ void kernelPickData( uchar *data, cdouble *bufs, cint offseti, cint offsetj, cint offsetk )
{
	GetIndex3D();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	double value = bufs[ Index(i, j, k) ];

	/* append data to volume data */
	int temp = atomicRound( value );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (uchar) temp;
};

__global__ void kernelCopyGrids( double *src, cdouble *dst )
{
	GetIndex3D();

	src[Index(i,j,k)] = dst[Index(i,j,k)];
};

__global__ void kernelInteractNodes( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back )
{
	GetIndex3D();

    if ( up[Index(i,sim_header,k)] )
		center[Index(i,sim_tailer,k)] = ( center[Index(i,sim_tailer,k)] + up[Index(i,sim_header,k)] ) / 2.f;

    if ( down[Index(i,sim_tailer,k)] )
        center[Index(i,sim_header,k)] = ( down[Index(i,sim_tailer,k)] + center[Index(i,sim_header,k)] ) / 2.f;

    if ( left[Index(sim_tailer,j,k)] )
        center[Index(sim_header,j,k)] = ( center[Index(sim_header,j,k)] + left[Index(sim_tailer,j,k)] ) / 2.f;

	if ( right[Index(sim_header,j,k)] )
		center[Index(sim_tailer,j,k)] = ( center[Index(sim_tailer,j,k)] + right[Index(sim_header,j,k)] ) / 2.f;

    if ( front[Index(i,j,sim_header)] )
        center[Index(i,j,sim_tailer)] = ( front[Index(i,j,sim_header)] + center[Index(i,j,sim_tailer)] ) / 2.f;

    if ( back[Index(i,j,sim_tailer)] )
        center[Index(i,j,sim_header)] = ( center[Index(i,j,sim_header)] + back[Index(i,j,sim_tailer)] ) / 2.f;
};

__global__ void kernelSumDensity( int *tpbuf, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back )
{
	GetIndex3D();
	if ( tpbuf[MACRO_LEFT] eqt MACRO_FALSE and left[Index(i,j,k)] > 0.f )
		tpbuf[MACRO_LEFT] = MACRO_TRUE;

	if ( tpbuf[MACRO_RIGHT] eqt MACRO_FALSE and right[Index(i,j,k)] > 0.f )	
		tpbuf[MACRO_RIGHT] = MACRO_TRUE;
 
	if ( tpbuf[MACRO_UP] eqt MACRO_FALSE and up[Index(i,j,k)] > 0.f )
		tpbuf[MACRO_UP] = MACRO_TRUE;
 
	if ( tpbuf[MACRO_DOWN] eqt MACRO_FALSE and down[Index(i,j,k)] > 0.f )
		tpbuf[MACRO_DOWN] = MACRO_TRUE;
 
	if ( tpbuf[MACRO_FRONT] eqt MACRO_FALSE and front[Index(i,j,k)] > 0.f )
		tpbuf[MACRO_FRONT] = MACRO_TRUE;

	if ( tpbuf[MACRO_BACK] eqt MACRO_FALSE and back[Index(i,j,k)] > 0.f )
		tpbuf[MACRO_BACK] = MACRO_TRUE;
};

#endif