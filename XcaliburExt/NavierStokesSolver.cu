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

void FluidSimProc::SolveNavierStokesEquation( cdouble timestep, bool add )
{
	if ( add ) SourceSolver();
	VelocitySolver( timestep );
	DensitySolver( timestep );
};

void FluidSimProc::SourceSolver( void )
{
	Dim3ParamDim();

	kernelAddSource __device_func__ ( dev_den, dev_u, dev_v, dev_w );
};


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
	Advection( dev_u0, dev_u, timestep, dev_u, dev_v, dev_w );
	Advection( dev_v0, dev_v, timestep, dev_u, dev_v, dev_w );
	Advection( dev_w0, dev_w, timestep, dev_u, dev_v, dev_w );

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
	Advection ( dev_den, dev_den0, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	Dim3ParamDim();

	for ( int k=0; k<20; k++)
	{
		kernelJacobi __device_func__ ( out, in, diff, divisor );
	}
};

void FluidSimProc::Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
	Dim3ParamDim();
	kernelAdvection __device_func__ ( out, in, timestep, u, v, w );
};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_X * GRIDS_X;
	Jacobi ( out, in, rate, 1+6*rate );
};

void FluidSimProc::Projection( double *u, double *v, double *w, double *div, double *p )
{
	Dim3ParamDim();

	// the velocity gradient
	kernelGradient __device_func__ ( div, p, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract __device_func__ ( u, v, w, p );
};