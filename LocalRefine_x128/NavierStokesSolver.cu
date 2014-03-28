/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 24, 2014
* <File Name>     NavierStokesSolver.cu
*/

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "Kernels.h"

using namespace sge;

void FluidSimProc::SourceSolverGlobal( cdouble dt )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelAddSource __device_func__
			( dev_den, dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, 
			dev_obs, dt, (double)(rand() % 300 + 1) / 100.f );
};


void FluidSimProc::VelocitySolverGlobal( cdouble dt )
{
	// diffuse the velocity field (per axis):
	DiffusionGlobal( dev_u0, dev_u, VISOCITY );
	DiffusionGlobal( dev_v0, dev_v, VISOCITY );
	DiffusionGlobal( dev_w0, dev_w, VISOCITY );
	
	if ( m_scHelper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	ProjectionGlobal( dev_u, dev_v, dev_w, dev_div, dev_p );

	if ( m_scHelper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	AdvectionGlobal( dev_u0, dev_u, dt, dev_u, dev_v, dev_w );
	AdvectionGlobal( dev_v0, dev_v, dt, dev_u, dev_v, dev_w );
	AdvectionGlobal( dev_w0, dev_w, dt, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	ProjectionGlobal( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolverGlobal( cdouble dt )
{
	DiffusionGlobal( dev_den0, dev_den, DIFFUSION );
	std::swap( dev_den0, dev_den );
	AdvectionGlobal( dev_den, dev_den0, dt, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::JacobiGlobal( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k=0; k<20; k++)
	{
		kernelJacobi __device_func__ ( out, in, GRIDS_X, GRIDS_Y, GRIDS_Z, diff, divisor );
	}
};

void FluidSimProc::AdvectionGlobal( double *out, cdouble *in, cdouble dt, cdouble *u, cdouble *v, cdouble *w )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
	
	kernelAdvection __device_func__ ( out, in, GRIDS_X, GRIDS_Y, GRIDS_Z, dt, u, v, w );
};

void FluidSimProc::DiffusionGlobal( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_X * GRIDS_X;
	JacobiGlobal( out, in, rate, 1+6*rate );
};

void FluidSimProc::ProjectionGlobal( double *u, double *v, double *w, double *div, double *p )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// the velocity gradient
	kernelGradient __device_func__ ( div, p, GRIDS_X, GRIDS_Y, GRIDS_Z, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	JacobiGlobal(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract __device_func__ ( u, v, w, p, GRIDS_X, GRIDS_Y, GRIDS_Z );
};


//////////////////////////////////////////////////////////////////////////////////////////////


void FluidSimProc::VelocitySolverLocal( cdouble dt )
{
	// diffuse the velocity field (per axis):
	DiffusionLocal( loc_u0, loc_u, VISOCITY );
	DiffusionLocal( loc_v0, loc_v, VISOCITY );
	DiffusionLocal( loc_w0, loc_w, VISOCITY );
	
	if ( m_scHelper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( loc_u0, loc_u );
	std::swap( loc_v0, loc_v );
	std::swap( loc_w0, loc_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	ProjectionLocal( loc_u, loc_v, loc_w, loc_div, loc_p );

	if ( m_scHelper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	AdvectionLocal( loc_u0, loc_u, dt, loc_u, loc_v, loc_w );
	AdvectionLocal( loc_v0, loc_v, dt, loc_u, loc_v, loc_w );
	AdvectionLocal( loc_w0, loc_w, dt, loc_u, loc_v, loc_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( loc_u0, loc_u );
	std::swap( loc_v0, loc_v );
	std::swap( loc_w0, loc_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	ProjectionGlobal( loc_u, loc_v, loc_w, loc_div, loc_p );
};

void FluidSimProc::DensitySolverLocal( cdouble dt )
{
	DiffusionLocal( loc_den0, loc_den, DIFFUSION );
	std::swap( loc_den0, loc_den );
	AdvectionLocal( loc_den, loc_den0, dt, loc_u, loc_v, loc_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::JacobiLocal( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k = 0; k < 20; k++)
		kernelJacobi __device_func__ ( out, in, GRIDS_X, GRIDS_Y, GRIDS_Z, diff, divisor );
};

void FluidSimProc::AdvectionLocal( double *out, cdouble *in, cdouble dt, cdouble *u, cdouble *v, cdouble *w )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
	
	kernelAdvection __device_func__ ( out, in, GRIDS_X, GRIDS_Y, GRIDS_Z, dt, u, v, w );
};

void FluidSimProc::DiffusionLocal( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_X * GRIDS_X;
	JacobiGlobal( out, in, rate, 1+6*rate );
};

void FluidSimProc::ProjectionLocal( double *u, double *v, double *w, double *div, double *p )
{
	m_scHelper.DeviceParamDim
		( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// the velocity gradient
	kernelGradient __device_func__ ( div, p, GRIDS_X, GRIDS_Y, GRIDS_Z, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	JacobiGlobal(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract __device_func__ ( u, v, w, p, GRIDS_X, GRIDS_Y, GRIDS_Z );
};