/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Feb 23, 2014
* <File Name>     FluidSimulationKernels.h
*/

#ifndef __fluid_simulation_kernels_h__
#define __fluid_simulation_kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "CUDACustomeMath.h"
#include "MacroDefinition.h"
#include "CUDAMacroDef.h"
#include "FluidSimProc.h"

inline __host__ __device__  double atomicGetValue
	( double const *grid, int const x, int const y, int const z )
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_tailer ) return 0.f;
	if ( y > gst_tailer ) return 0.f;
	if ( z > gst_tailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

inline __host__ __device__  void atomicVertices
	( double *c000, double *c001, double *c011, double *c010,
	double *c100, double *c101, double *c111, double *c110,
	double const *grid, double const x, double const y, double const z )
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

inline __host__ __device__  double atomicTrilinear
	( double const *grid, double const x, double const y, double const z )
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

__device__ void atomicDensityObs( double *grids, const double *obstacle )
{
	GetIndex();
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

__device__ void atomicVelocityObs_U( double *grids, const double *obstacle )
{
	GetIndex();
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

__device__ void atomicVelocityObs_V( double *grids, const double *obstacle )
{
	GetIndex();
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

__device__ void atomicVelocityObs_W( double *grids, const double *obstacle )
{
	GetIndex();
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

__global__ void kernelObstacle( double *grids, const double *obstacle, const int field )
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

__global__ void kernelJacobi( double *grid_out, double const *grid_in, double const diffusion, double const divisor )
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

__host__ void hostJacobi
	( double *grid_out, double const *grid_in,
	double const *obstacle, int const field, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim) (grid_out, grid_in, diffusion, divisor);
	}
	kernelObstacle cudaDevice(gridDim, blockDim) ( grid_out, obstacle, field );
};

__global__ void kernelGridAdvection( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;
	
	grid_out [ Index(i,j,k) ] = atomicTrilinear ( grid_in, u, v, w );

	EndSimArea();
};

__host__ void hostAdvection
	( double *grid_out, double const *grid_in,
	double const *obstacle, int const field,
	double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim) ( grid_out, grid_in, u_in, v_in, w_in );
	kernelObstacle cudaDevice(gridDim, blockDim) ( grid_out, obstacle, field );
};

__host__ void hostDiffusion
	( double *grid_out, double const *grid_in, double const diffusion,
	double const *obstacle, int const field )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, obstacle, field, rate, 1+6*rate );
};


__global__ void kernelGradient( double *div, double *p, double const *vel_u, double const *vel_v, double const *vel_w )
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

__host__ void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p, double const *obs )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) ( div, p, vel_u, vel_v, vel_w );
	kernelObstacle cudaDevice(gridDim, blockDim) ( div, obs, MACRO_SIMPLE );
	kernelObstacle cudaDevice(gridDim, blockDim) ( p, obs, MACRO_SIMPLE );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, obs, MACRO_SIMPLE, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) ( vel_u, vel_v, vel_w, p );
	kernelObstacle cudaDevice(gridDim, blockDim) ( vel_u, obs, MACRO_VELOCITY_U );
	kernelObstacle cudaDevice(gridDim, blockDim) ( vel_v, obs, MACRO_VELOCITY_V );
	kernelObstacle cudaDevice(gridDim, blockDim) ( vel_w, obs, MACRO_VELOCITY_W );
};

__global__ void kernelSetBoundary( double *grids )
{
	GetIndex();
	
	const int half = GRIDS_X / 2;
	
	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
		grids[ Index(i,j,k) ] = MACRO_BOUNDARY_SOURCE;
};

__global__ void kernelAddSource
	( double *density, double *vel_u, double *vel_v, double *vel_w, double *obs )
{
	GetIndex();

	const int half = GRIDS_X / 2;

	if ( obs[ Index(i,j,k) ] eqt MACRO_BOUNDARY_SOURCE )
	{
		/* add source to grids */
		density[Index(i,j,k)] = SOURCE_DENSITY;

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
	}
};

#endif