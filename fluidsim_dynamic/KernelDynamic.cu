/**
* <Author>      Orlando Chen
* <First>       Jan 10, 2014
* <Last>		Jan 10, 2014
* <File>        kernel.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "FluidSimAreaDynamic.h"
#include "FluidMathLibDynamic.h"
#include "OperationsDynamic.h"
#include "FunctionHelperDynamic.h"


using namespace sge;
using namespace std;

#pragma region void hostAddSource ( double *dens, double *vel_u, double *vel_v, double *vel_w  )
/** 
* number:
* 0 ------ add density
* 1 ------ add velocity v
*/
__global__ 	
void kernelAddSource ( double *grid, int const number )
{
	GetIndex();
	BeginSimArea();

	const int half = Grids_X / 2;

	switch ( number )
	{
	case 0: // density
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = VOLUME;

	case 1: // velocity v
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = VOLUME * 2.f;

	default: // add external force if need
		break;
	}

	EndSimArea();
};

__host__
void hostAddSource ( double *dens, double *vel_u, double *vel_v, double *vel_w  )
{
	cudaDeviceDim3D();

	if ( dens != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( dens, 0 );
	if ( vel_v != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( vel_v, 1 );
};
#pragma endregion


/**
* cd:
* 0 -------- solve density
* 1 -------- solve velocity u
* 2 -------- solve velocity v
* 3 -------- solve velocity w
*/
__global__ void kernelBoundary ( double *grid, int const cd )
{
	GetIndex();
	BeginSimArea();
	// ...
	EndSimArea();
};


#pragma region void hostJacobi ( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
__global__
void kernelJacobi ( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
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

__host__
void hostJacobi ( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim) (grid_out, grid_in, cd, diffusion, divisor);
		kernelBoundary cudaDevice(gridDim, blockDim) (grid_out, cd);
	}
};
#pragma endregion


#pragma region void hostAdvection ( double *grid_out, double const *grid_in, int const cd, double const *u_in, double const *v_in, double const *w_in )
__global__ 
void kernelGridAdvection ( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTA_TIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTA_TIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTA_TIME;
	grid_out [ Index(i,j,k) ] = trilinear ( grid_in, u, v, w );

	EndSimArea();
};

__host__
void hostAdvection ( double *grid_out, double const *grid_in, int const cd, double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim) ( grid_out, grid_in, u_in, v_in, w_in );
	kernelBoundary cudaDevice(gridDim, blockDim) ( grid_out, cd );

};
#pragma endregion


__host__ void hostDiffusion ( double *grid_out,
	double const *grid_in, int const cd, double const diffusion )
{
//	double rate = diffusion * Grids_X * Grids_X * Grids_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, cd, rate, 1+6*rate );
};


#pragma region void hostProject ( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
__global__
void kernelGradient ( double *div, double *p,
	double const *vel_u, double const *vel_v, double const *vel_w )
{
	GetIndex();
	BeginSimArea();
	
	const double h = 1.f / Grids_X;

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

__global__
void kernelSubtract ( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * Grids_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * Grids_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * Grids_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__
void hostProject ( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) (div, p, vel_u, vel_v, vel_w);
	kernelBoundary cudaDevice(gridDim, blockDim) (div, 0);
	kernelBoundary cudaDevice(gridDim, blockDim) (p, 0);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 0, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) (vel_u, vel_v, vel_w, p);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_u, 1);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_v, 2);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_w, 3);
};
#pragma endregion


__global__ 
void kernelReplaceData ( unsigned char* origin, unsigned const char* small,
	int const offx, int const offy, int const offz )
{
	GetIndex();

	int index = Index( i + offx * Grids_X, j + offy * Grids_X, k + offz * Grids_X );
	origin[index] = small[Index(i,j,k)];
}


#pragma region velocity, density, fluid simulation solver and pick data

void FluidSimProc::VelocitySolver ( void )
{
	hostAddSource ( NULL, NULL, dev_v, NULL );

	// diffuse the velocity field (per axis):
	hostDiffusion ( dev_0, dev_u, 1, VISOCITY );
	hostDiffusion ( dev_1, dev_v, 2, VISOCITY );
	hostDiffusion ( dev_2, dev_w, 3, VISOCITY );
	hostSwapBuffer ( dev_0, dev_u );
	hostSwapBuffer ( dev_1, dev_v );
	hostSwapBuffer ( dev_2, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject ( dev_u, dev_v, dev_w, dev_div, dev_p );
	
	// advect the velocity field (per axis):
	hostAdvection ( dev_0, dev_u, 1, dev_u, dev_v, dev_w );
	hostAdvection ( dev_1, dev_v, 2, dev_u, dev_v, dev_w );
	hostAdvection ( dev_2, dev_w, 3, dev_u, dev_v, dev_w );
	hostSwapBuffer ( dev_0, dev_u );
	hostSwapBuffer ( dev_1, dev_v );
	hostSwapBuffer ( dev_2, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject ( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolver ( void )
{
	hostAddSource ( dev_den, NULL, NULL, NULL );
	hostDiffusion ( dev_0, dev_den, 0, DIFFUSION );
	hostSwapBuffer ( dev_0, dev_den );
	hostAdvection ( dev_den, dev_0, 0, dev_u, dev_v, dev_w );
};


void FluidSimProc::PickData ( fluidsim *fluid )
{
	cudaDeviceDim3D ();
	kernelPickData  <<<gridDim, blockDim>>> ( dev_smallv, dev_den );
	kernelReplaceData <<<gridDim, blockDim>>> ( dev_visual, dev_smallv, IXi, IXj, IXk );

	size_t size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;
	if ( cudaMemcpy (host_visual, dev_visual, 
		sizeof(uchar) * size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed", __FILE__, __LINE__);
		FreeResourcePtrs ();
		exit (1);
	}
};


void FluidSimProc::FluidSimSolver ( fluidsim *fluid )
{
	if ( !fluid->ray.bRun ) return ;
	
	// Zero buffer first
	cudaDeviceDim3D();
	for ( int i = 0; i < DevListNum; i++ )
	{
		kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_list[i] );
	}

	// For fluid simulation, copy the data to device
	CopyDataToDevice();

	// Fluid process
	VelocitySolver ();
	DensitySolver ();
	PickData ( fluid );

	// Synchronize the device
	if ( cudaDeviceSynchronize() != cudaSuccess ) goto Error;

	// After simulation process, retrieve data back to host, in order to 
	// avoid data flipping
	CopyDataToHost();

	goto Success;

Error:
	cudaCheckErrors ("cudaDeviceSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	fluid->volume.ptrData = host_visual;
};

#pragma endregion