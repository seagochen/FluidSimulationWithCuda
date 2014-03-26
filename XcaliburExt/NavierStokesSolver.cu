/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 24, 2014
* <File Name>     NavierStokesSolver.cu
*/

#include <ctime>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "Kernels.h"

using namespace sge;
using namespace std;

static FunctionHelper m_helper;
static dim3 grid, block;

#define __device_func__ <<<grid,block>>>

void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_helper.DeviceParamDim( &grid, &block, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k=0; k<20; k++)
	{
		kernelJacobi __device_func__ ( out, in, diff, divisor );
	}
};

void LocalAdvection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
	m_helper.DeviceParamDim( &grid, &block, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelAdvection __device_func__ ( out, in, timestep, u, v, w );
};

void Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_X * GRIDS_X;
	Jacobi ( out, in, rate, 1+6*rate );
};

void Projection( double *u, double *v, double *w, double *div, double *p )
{
	m_helper.DeviceParamDim( &grid, &block, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// the velocity gradient
	kernelGradient __device_func__ ( div, p, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract __device_func__ ( u, v, w, p );
};

#if 0

#define dev_u m_vectGPUVelU[0]
#define dev_v m_vectGPUVelV[0]
#define dev_w m_vectGPUVelW[0]
#define dev_den m_vectGPUDens[0]

#define dev_div m_vectGPUDiv[0]

#define dev_p m_vectGPUPress[0]

#define dev_u0 m_vectGPUVelU[1]
#define dev_v0 m_vectGPUVelV[1]
#define dev_w0 m_vectGPUVelW[1]
#define dev_den0 m_vectGPUDens[1]



void FluidSimProc::VelocitySolver( cdouble timestep )
{
	// diffuse the velocity field (per axis):
	Diffusion( dev_u0, dev_u, VISOCITY );
	Diffusion( dev_v0, dev_v, VISOCITY );
	Diffusion( dev_w0, dev_w, VISOCITY );
	
	if ( m_scHelper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );

	if ( m_scHelper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	LocalAdvection( dev_u0, dev_u, timestep, dev_u, dev_v, dev_w );
	LocalAdvection( dev_v0, dev_v, timestep, dev_u, dev_v, dev_w );
	LocalAdvection( dev_w0, dev_w, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolver( cdouble timestep )
{
	Diffusion( dev_den0, dev_den, DIFFUSION );
	std::swap( dev_den0, dev_den );
	LocalAdvection ( dev_den, dev_den0, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::SourceSolver( cdouble dt )
{
	GridsParamDim();
	kernelAddSource __device_func__
			( dev_den, dev_v, m_vectGPUObst[0], dt, (double)(rand() % 300 + 1) / 100.f );
}

#else

void FluidSimProc::SourceSolver( cdouble dt )
{
	GridsParamDim();
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
		kernelAddSource __device_func__
			( m_vectGPUDens[i], m_vectGPUVelV[i], m_vectGPUObst[i], dt, (double)(rand() % 300 + 1) / 100.f );
	
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		kernelExitBullet __device_func__
			( m_vectDevDens[i], m_vectGPUDens[i], GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
		kernelExitBullet __device_func__
			( m_vectDevVelV[i], m_vectGPUVelV[i], GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	}
	
	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelInterLeafGrids __device_func__ 
			( global_dens, m_vectDevDens[ix(i,j,k,NODES_X,NODES_Y)], i, j, k, 0.5f );
		kernelInterLeafGrids __device_func__ 
			( global_velv, m_vectDevVelV[ix(i,j,k,NODES_X,NODES_Y)], i, j, k, 0.5f );
	}

	if ( m_scHelper.GetCUDALastError( "call member function SourceSolver failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::VelocitySolver( cdouble dt )
{
};

void FluidSimProc::DensitySolver( cdouble dt )
{
};

#endif