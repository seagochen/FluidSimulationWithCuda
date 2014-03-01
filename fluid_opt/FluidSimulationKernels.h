/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 01, 2014
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

__device__  double atomicGetValue( double const *grid, int const x, int const y, int const z )
{
	if ( x < 0 or x > 65 )  return 0.f;
	if ( y < 0 or y > 65 )  return 0.f;
	if ( z < 0 or z > 65 )  return 0.f;

	return grid[ Index3D(x,y,z,SLOT_X) ];
};

__device__  void atomicVertices
	( double *c000, double *c001, double *c011, double *c010, double *c100, 
	double *c101, double *c111, double *c110, double const *grid, double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	*c000 = atomicGetValue( grid, i, j, k );
	*c001 = atomicGetValue( grid, i, j+1, k );
	*c011 = atomicGetValue( grid, i, j+1, k+1 );
	*c010 = atomicGetValue( grid, i, j, k+1 );
	*c100 = atomicGetValue( grid, i+1, j, k );
	*c101 = atomicGetValue( grid, i+1, j+1, k );
	*c111 = atomicGetValue( grid, i+1, j+1, k+1 );
	*c110 = atomicGetValue( grid, i+1, j, k+1 );
}

__device__  double atomicTrilinear( double const *grid, double const x, double const y, double const z )
{
	double v000, v001, v010, v011, v100, v101, v110, v111;
	
	atomicVertices( &v000, &v001, &v011, &v010, &v100, &v101, &v111, &v110, grid, x, y, z );

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

	i = i + 1;
	j = j + 1;
	k = k + 1;

	/* 当前格点有障碍物，且密度大于0 */
	if ( obstacle[Index3D(i,j,k,SLOT_X)] eqt MACRO_BOUNDARY_OBSTACLE and grids[Index3D(i,j,k,SLOT_X)] > 0.f )
	{
		int cells  = 0;
		double val = 0; 

		if ( obstacle[Index3D(i-1,j,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index3D(i+1,j,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index3D(i,j-1,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index3D(i,j+1,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index3D(i,j,k-1,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index3D(i,j,k+1,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) cells++;

		if ( cells > 0 ) val = grids[Index3D(i,j,k,SLOT_X)] / cells;
		else val = 0.f;

		if ( obstacle[Index3D(i-1,j,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) grids[Index3D(i-1,j,k,SLOT_X)] += val;
		if ( obstacle[Index3D(i+1,j,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) grids[Index3D(i+1,j,k,SLOT_X)] += val;
		if ( obstacle[Index3D(i,j-1,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) grids[Index3D(i,j-1,k,SLOT_X)] += val;
		if ( obstacle[Index3D(i,j+1,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) grids[Index3D(i,j+1,k,SLOT_X)] += val;
		if ( obstacle[Index3D(i,j,k-1,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) grids[Index3D(i,j,k-1,SLOT_X)] += val;
		if ( obstacle[Index3D(i,j,k+1,SLOT_X)] eqt MACRO_BOUNDARY_BLANK ) grids[Index3D(i,j,k+1,SLOT_X)] += val;

		grids[Index3D(i,j,k,SLOT_X)] = 0.f;
	}
};

__device__ void atomicVelocityObs_U( double *grids, const double *obstacle )
{
	GetIndex();

	i = i + 1;
	j = j + 1;
	k = k + 1;

	if ( obstacle[Index3D(i,j,k,SLOT_X)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index3D(i,j,k,SLOT_X)] > 0.f )
		{
			if ( obstacle[Index3D(i-1,j,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index3D(i-1,j,k,SLOT_X)] = grids[Index3D(i-1,j,k,SLOT_X)] -  grids[Index3D(i,j,k,SLOT_X)];
		}
		else
		{
			if ( obstacle[Index3D(i+1,j,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index3D(i+1,j,k,SLOT_X)] = grids[Index3D(i+1,j,k,SLOT_X)] -  grids[Index3D(i,j,k,SLOT_X)];
		}
		grids[Index3D(i,j,k,SLOT_X)] = 0.f;
	}
};

__device__ void atomicVelocityObs_V( double *grids, const double *obstacle )
{
	GetIndex();

	i = i + 1;
	j = j + 1;
	k = k + 1;

	if ( obstacle[Index3D(i,j,k,SLOT_X)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index3D(i,j,k,SLOT_X)] > 0.f )
		{
			if ( obstacle[Index3D(i,j-1,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index3D(i,j-1,k,SLOT_X)] = grids[Index3D(i,j-1,k,SLOT_X)] - grids[Index3D(i,j,k,SLOT_X)];
		}
		else
		{
			if ( obstacle[Index3D(i,j+1,k,SLOT_X)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index3D(i,j+1,k,SLOT_X)] = grids[Index3D(i,j+1,k,SLOT_X)] - grids[Index3D(i,j,k,SLOT_X)];
		}
		grids[Index3D(i,j,k,SLOT_X)] = 0.f;
	}
};

__device__ void atomicVelocityObs_W( double *grids, const double *obstacle )
{
	GetIndex();

	i = i + 1;
	j = j + 1;
	k = k + 1;

	if ( obstacle[Index3D(i,j,k,SLOT_X)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index3D(i,j,k,SLOT_X)] > 0.f )
		{
			if ( obstacle[Index3D(i,j,k-1,SLOT_X)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index3D(i,j,k-1,SLOT_X)] = grids[Index3D(i,j,k-1,SLOT_X)] - grids[Index3D(i,j,k,SLOT_X)];
		}
		else
		{
			if ( obstacle[Index3D(i,j,k+1,SLOT_X)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index3D(i,j,k+1,SLOT_X)] = grids[Index3D(i,j,k+1,SLOT_X)] - grids[Index3D(i,j,k,SLOT_X)];
		}
		grids[Index3D(i,j,k,SLOT_X)] = 0.f;
	}
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

__global__ void kernelJacobi
	( double *grid_out, double const *grid_in, double const diffusion, double const divisor )
{
	GetIndex();

	i = i + 1;
	j = j + 1;
	k = k + 1;

	double div = 0.f;
	if ( divisor <= 0.f ) div = 1.f;
	else div = divisor;

	grid_out [ Index3D(i,j,k,SLOT_X) ] = 
		( grid_in [ Index3D(i,j,k,SLOT_X) ] + diffusion * 
			(
				grid_out [ Index3D(i-1, j, k, SLOT_X) ] + grid_out [ Index3D(i+1, j, k, SLOT_X) ] +
				grid_out [ Index3D(i, j-1, k, SLOT_X) ] + grid_out [ Index3D(i, j+1, k, SLOT_X) ] +
				grid_out [ Index3D(i, j, k-1, SLOT_X) ] + grid_out [ Index3D(i, j, k+1, SLOT_X) ]
			) 
		) / div;
}

__global__ void kernelGridAdvection
	( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();

	i = i + 1;
	j = j + 1;
	k = k + 1;

	double u = i - u_in [ Index3D(i,j,k,SLOT_X) ] * DELTATIME;
	double v = j - v_in [ Index3D(i,j,k,SLOT_X) ] * DELTATIME;
	double w = k - w_in [ Index3D(i,j,k,SLOT_X) ] * DELTATIME;
	
	grid_out [ Index3D(i,j,k,SLOT_X) ] = atomicTrilinear ( grid_in, u, v, w );
};

__global__ void kernelGradient
	( double *div, double *p, double const *vel_u, double const *vel_v, double const *vel_w )
{
	GetIndex();
	i = i + 1;
	j = j + 1;
	k = k + 1;

	const double h = 1.f / GRID_X;

	// previous instantaneous magnitude of velocity gradient 
	//		= (sum of velocity gradients per axis)/2N:
	div [ Index3D(i,j,k,SLOT_X) ] = -0.5f * h * (
			vel_u [ Index3D(i+1, j, k, SLOT_X) ] - vel_u [ Index3D(i-1, j, k, SLOT_X) ] + // gradient of u
			vel_v [ Index3D(i, j+1, k, SLOT_X) ] - vel_v [ Index3D(i, j-1, k, SLOT_X) ] + // gradient of v
			vel_w [ Index3D(i, j, k+1, SLOT_X) ] - vel_w [ Index3D(i, j, k-1, SLOT_X) ]   // gradient of w
		);
	// zero out the present velocity gradient
	p [ Index3D(i,j,k,SLOT_X) ] = 0.f;
};

__global__ void kernelSubtract( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	i = i + 1;
	j = j + 1;
	k = k + 1;
	// gradient calculated by neighbors

	vel_u [ Index3D(i, j, k, SLOT_X) ] -= 0.5f * GRID_X * ( p [ Index3D(i+1, j, k, SLOT_X) ] - p [ Index3D(i-1, j, k, SLOT_X) ] );
	vel_v [ Index3D(i, j, k, SLOT_X) ] -= 0.5f * GRID_X * ( p [ Index3D(i, j+1, k, SLOT_X) ] - p [ Index3D(i, j-1, k, SLOT_X) ] );
	vel_w [ Index3D(i, j, k, SLOT_X) ] -= 0.5f * GRID_X * ( p [ Index3D(i, j, k+1, SLOT_X) ] - p [ Index3D(i, j, k-1, SLOT_X) ] );
};

__global__ void kernelSetBoundary( double *grids )
{
	GetIndex();

	i = i + 1;
	j = j + 1;
	k = k + 1;
	
	const int half = GRID_X / 2;

#if !TESTING_MODE_SWITCH
	
	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
		grids[ Index3D(i,j,k,SLOT_X) ] = MACRO_BOUNDARY_SOURCE;
#else
	if ( i >= half-2 and i < half+2 and
		j >= half-2 and j < half+2 and
		k >= half-2 and k < half+2 )
		grids[Index3D(i,j,k,SLOT_X)] = MACRO_BOUNDARY_SOURCE;
#endif
};

__global__ void kernelAddSource
	( double *density, double *vel_u, double *vel_v, double *vel_w, double *obs, double const times )
{
	GetIndex();
	i = i + 1;
	j = j + 1;
	k = k + 1;

	if ( obs[ Index3D(i,j,k,SLOT_X) ] eqt MACRO_BOUNDARY_SOURCE )
	{
		/* add source to grids */
		density[Index3D(i,j,k,SLOT_X)] = SOURCE_DENSITY;

#if !TESTING_MODE_SWITCH
	const int half = GRID_X / 2;
	
	/* add velocity to grids */
	if ( i < half )
		vel_u[Index3D(i,j,k,SLOT_X)] = -SOURCE_VELOCITY * DELTATIME * DELTATIME;
	elif( i >= half )
		vel_u[Index3D(i,j,k,SLOT_X)] =  SOURCE_VELOCITY * DELTATIME * DELTATIME;
	
	vel_v[Index3D(i,j,k,SLOT_X)] = SOURCE_VELOCITY * times;
	
	if ( k < half )
		vel_w[Index3D(i,j,k,SLOT_X)] = -SOURCE_VELOCITY * DELTATIME * DELTATIME;
	elif ( k >= half )
		vel_w[Index3D(i,j,k,SLOT_X)] =  SOURCE_VELOCITY * DELTATIME * DELTATIME;
#else

	/* velocity: default-up(0) down(1) left(2) right(3) front(4) back(5) */
#if TESTING_MODE==0
		vel_v[Index3D(i,j,k,SLOT_X)] =  SOURCE_VELOCITY * times;
#elif TESTING_MODE==1
		vel_v[Index3D(i,j,k,SLOT_X)] = -SOURCE_VELOCITY * times;
#elif TESTING_MODE==2
		vel_u[Index3D(i,j,k,SLOT_X)] = -SOURCE_VELOCITY * times;
#elif TESTING_MODE==3
		vel_u[Index3D(i,j,k,SLOT_X)] =  SOURCE_VELOCITY * times;
#elif TESTING_MODE==4
		vel_w[Index3D(i,j,k,SLOT_X)] =  SOURCE_VELOCITY * times;
#elif TESTING_MODE==5
		vel_w[Index3D(i,j,k,SLOT_X)] = -SOURCE_VELOCITY * times;
#endif

#endif
	}
};

void hostJacobi
	( double *grid_out, double const *grid_in, double const *obstacle, 
	int const field, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi <<<gridDim, blockDim>>> (grid_out, grid_in, diffusion, divisor);
	}
	kernelObstacle <<<gridDim, blockDim>>> ( grid_out, obstacle, field );
};

void hostAdvection
	( double *grid_out, double const *grid_in, double const *obstacle, int const field, 
	double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection <<<gridDim, blockDim>>> ( grid_out, grid_in, u_in, v_in, w_in );
	kernelObstacle <<<gridDim, blockDim>>> ( grid_out, obstacle, field );
};

void hostDiffusion
	( double *grid_out, double const *grid_in, double const diffusion, double const *obstacle, int const field )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi( grid_out, grid_in, obstacle, field, rate, 1+6*rate );
};

void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p, double const *obs )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient <<<gridDim, blockDim>>> ( div, p, vel_u, vel_v, vel_w );
	kernelObstacle <<<gridDim, blockDim>>> ( div, obs, MACRO_SIMPLE );
	kernelObstacle <<<gridDim, blockDim>>> ( p, obs, MACRO_SIMPLE );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, obs, MACRO_SIMPLE, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract <<<gridDim, blockDim>>> ( vel_u, vel_v, vel_w, p );
	kernelObstacle <<<gridDim, blockDim>>> ( vel_u, obs, MACRO_VELOCITY_U );
	kernelObstacle <<<gridDim, blockDim>>> ( vel_v, obs, MACRO_VELOCITY_V );
	kernelObstacle <<<gridDim, blockDim>>> ( vel_w, obs, MACRO_VELOCITY_W );
};

#endif