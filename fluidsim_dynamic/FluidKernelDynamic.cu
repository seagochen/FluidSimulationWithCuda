/**
* <Author>      Orlando Chen
* <First>       Dec 12, 2013
* <Last>		Jan 13, 2013
* <File>        FluidKernelDynamic.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "FluidSimAreaDynamic.h"
#include "FluidMathLibDynamic.h"
#include "BufferOperationDynamic.h"

using namespace sge;
using namespace std;

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

	const int half = GRIDS_X / 2;

	switch ( number )
	{
	case 0: // density
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = SOURCE;

	case 1: // velocity v
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = SOURCE;

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

/**
* face:
* 0 ------- up
* 1 ------- down 
* 2 ------- left
* 3 ------- right
* 4 ------- front
* 5 ------- back
*/
__device__ double checksum ( double const *grid, int const face )
{

};

/**
* cd:
* 0 -------- solve density
* 1 -------- solve velocity u
* 2 -------- solve velocity v
* 3 -------- solve velocity w
*/
__global__ void kernelBoundary ( double *grid,
	int const cd,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	GetIndex();
	BeginSimArea();
	// ...
	EndSimArea();
};

__global__
void kernelJacobi ( double *grid_out,
	double const *grid_in, 
	int const cd, double const diffusion, double const divisor )
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
void hostJacobi ( double *grid_out, 
	double const *grid_in, 
	int const cd, double const diffusion, double const divisor,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	cudaDeviceDim3D();
	for ( int k = 0; k < 20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim)
			( grid_out, grid_in, cd, diffusion, divisor );
		kernelBoundary cudaDevice(gridDim, blockDim) 
			( grid_out, cd, up, down, left, right, front, back, obstacle );
	}
};

__global__ 
void kernelGridAdvection ( double *grid_out,
	double const *grid_in, 
	double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;
	grid_out [ Index(i,j,k) ] = trilinear ( grid_in, u, v, w );

	EndSimArea();
};

__host__
void hostAdvection ( double *grid_out,
	double const *grid_in, int const cd, 
	double const *u_in, double const *v_in, double const *w_in,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim)
		( grid_out, grid_in, u_in, v_in, w_in );
	kernelBoundary cudaDevice(gridDim, blockDim)
		( grid_out, cd, up, down, left, right, front, back, obstacle );

};

__host__ void hostDiffusion ( double *grid_out,
	double const *grid_in, int const cd, double const diffusion,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle 	)
{
	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	hostJacobi
		( grid_out, grid_in, cd, rate, 1 + 6 * rate, up, down, left, right, front, back, obstacle );
};

__global__
void kernelGradient ( double *div, double *p,
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

__global__
void kernelSubtract ( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__
void hostProject ( double *vel_u, double *vel_v, double *vel_w, double *div, double *p,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle  )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) 
		( div, p, vel_u, vel_v, vel_w );
	kernelBoundary cudaDevice(gridDim, blockDim)
		( div, 0, up, down ,left, right, front, back, obstacle );
	kernelBoundary cudaDevice(gridDim, blockDim)
		( p, 0, up, down ,left, right, front, back, obstacle );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 0, 1.f, 6.f, up, down ,left, right, front, back, obstacle );

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) 
		( vel_u, vel_v, vel_w, p );
	kernelBoundary cudaDevice(gridDim, blockDim)
		( vel_u, 1, up, down ,left, right, front, back, obstacle );
	kernelBoundary cudaDevice(gridDim, blockDim) 
		( vel_v, 2, up, down ,left, right, front, back, obstacle );
	kernelBoundary cudaDevice(gridDim, blockDim) 
		( vel_w, 3, up, down ,left, right, front, back, obstacle );
};

#include "FunctionHelperDynamic.h"

void FluidSimProc::VelocitySolver ( void )
{
	// diffuse the velocity field (per axis):
	hostDiffusion
		( dev_u0, dev_u, 1, VISOCITY, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostDiffusion
		( dev_v0, dev_v, 2, VISOCITY, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostDiffusion
		( dev_w0, dev_w, 3, VISOCITY, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostSwapBuffer
		( dev_u0, dev_u );
	hostSwapBuffer
		( dev_v0, dev_v );
	hostSwapBuffer
		( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject
		( dev_u, dev_v, dev_w, dev_div, dev_p, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	
	// advect the velocity field (per axis):
	hostAdvection 
		( dev_u0, dev_u, 1, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostAdvection
		( dev_v0, dev_v, 2, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostAdvection
		( dev_w0, dev_w, 3, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostSwapBuffer
		( dev_u0, dev_u );
	hostSwapBuffer
		( dev_v0, dev_v );
	hostSwapBuffer
		( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject
		( dev_u, dev_v, dev_w, dev_div,	dev_p, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
};

void FluidSimProc::DensitySolver ( void )
{
	hostDiffusion
		( dev_den0, dev_den, 0, DIFFUSION, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostSwapBuffer
		( dev_den0, dev_den );
	hostAdvection
		( dev_den, dev_den0, 0, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
};

void FluidSimProc::PickData ( fluidsim *fluid )
{
	cudaDeviceDim3D ();
	int offseti = node_list[ IX ].i * GRIDS_X;
	int offsetj = node_list[ IX ].j * GRIDS_X;
	int offsetk = node_list[ IX ].k * GRIDS_X;

	kernelPickData cudaDevice(gridDim, blockDim) 
		( dev_visual, dev_den, offseti, offsetj, offsetk );

	size_t size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;
	if ( cudaMemcpy (host_visual, dev_visual, sizeof(uchar) * size, 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed", __FILE__, __LINE__);
		FreeResourcePtrs ();
		exit (1);
	}
};

void FluidSimProc::FluidSimSolver ( fluidsim *fluid )
{
	if ( !fluid->ray.bRun ) return ;
	
	/* round robin if node is active */
	for ( int i = 0; i < node_list.size(); i++ )
	{
		/* active! */
		if ( node_list[i].bActive == true )
		{
			/* zero buffer first */
			ZeroDevData();
			
			/* for fluid simulation, copy the data to device */
			CopyDataToDevice();

			/* add source if current node is active */
			if ( i eqt 10 )
			hostAddSource ( dev_den, NULL, dev_v, NULL );
			
			/* fluid process */
			VelocitySolver ();
			DensitySolver ();
			PickData ( fluid );
			
			/* Synchronize the device */
			if ( cudaDeviceSynchronize() != cudaSuccess ) goto Error;		
			
			/* after simulation process, retrieve data back to host, in order to 
			* avoid data flipping 
			*/
			CopyDataToHost();
		}
	}
	goto Success;

Error:
	cudaCheckErrors ("cudaDeviceSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	fluid->volume.ptrData = host_visual;
};