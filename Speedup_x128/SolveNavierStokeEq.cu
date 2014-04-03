/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Apr 01, 2014
* <File Name>     SolveNavierStokeEq.cu
*/

#include <time.h>
#include <iostream>
#include <utility>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "Kernels.h"

using namespace sge;
using std::cout;
using std::endl;

void FluidSimProc::SolveNavierStokesEquation
			( cdouble dt, bool add, bool vel, bool den,
			cint tx, cint ty,
			cint gx, cint gy, cint gz,
			cint bx, cint by, cint bz )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, tx, ty, gx, gy, gz );

#if 0
	double *ptr;
	m_scHelper.CreateHostBuffers( gBULLET_X * gBULLET_Y * gBULLET_Z * sizeof(double), 1, &ptr );

//	cudaMemcpy( ptr, dev_den, gBULLET_X * gBULLET_Y * gBULLET_Z * sizeof(double), cudaMemcpyDeviceToHost );
//
//	for ( int j = 0; j < gBULLET_Y; j++ )
//	{
//		for ( int k = 0; k < gBULLET_Z; k++ )
//		{
//			for ( int i = 0; i < gBULLET_X; i++ )
//			{
//				printf( "%d ", (int)ptr[ix(i,j,k,gBULLET_X,gBULLET_Y)] );
//			}
//			printf("\n");
//		}
//		printf("\n");
//	}

	cout << "----------------------------------------------------------------------" << endl;

#endif

	if (add) SourceSolver( dt, bx, by, bz );
	if (vel) VelocitySolver( dt, bx, by, bz );
	if (den) DensitySolver( dt, bx, by, bz );


#if 0

	cudaMemcpy( ptr, dev_p, gBULLET_X * gBULLET_Y * gBULLET_Z * sizeof(double), cudaMemcpyDeviceToHost );

	for ( int j = 0; j < gBULLET_Y; j++ )
	{
		for ( int k = 0; k < gBULLET_Z; k++ )
		{
			for ( int i = 0; i < gBULLET_X; i++ )
			{
				printf( "%f ", ptr[ix(i,j,k,gBULLET_X,gBULLET_Y)] );
			}
			printf("\n");
		}
		printf("\n");
	}

	free(ptr);
	FreeResource();
	exit(0);

#endif
};


void FluidSimProc::SourceSolver( cdouble dt, cint bx, cint by, cint bz )
{
	kernelAddSource __device_func__
		( *dev_den, *dev_v, bx, by, bz, 
			*dev_obs, dt, (double)(rand() % 300 + 1) / 100.f );
};


void FluidSimProc::VelocitySolver( cdouble dt, cint bx, cint by, cint bz )
{
	// diffuse the velocity field (per axis):
	Diffusion( *dev_u0, *dev_u, VISOCITY, bx, by, bz );
	Diffusion( *dev_v0, *dev_v, VISOCITY, bx, by, bz );
	Diffusion( *dev_w0, *dev_w, VISOCITY, bx, by, bz );
	
	if ( m_scHelper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( *dev_u0, *dev_u );
	std::swap( *dev_v0, *dev_v );
	std::swap( *dev_w0, *dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( *dev_u, *dev_v, *dev_w, *dev_div, *dev_p, bx, by, bz );

	if ( m_scHelper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	Advection( *dev_u0, *dev_u, *dev_u, *dev_v, *dev_w, dt, bx, by, bz );
	Advection( *dev_v0, *dev_v, *dev_u, *dev_v, *dev_w, dt, bx, by, bz );
	Advection( *dev_w0, *dev_w, *dev_u, *dev_v, *dev_w, dt, bx, by, bz );

	if ( m_scHelper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( *dev_u0, *dev_u );
	std::swap( *dev_v0, *dev_v );
	std::swap( *dev_w0, *dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( *dev_u, *dev_v, *dev_w, *dev_div, *dev_p, bx, by, bz );
};


void FluidSimProc::DensitySolver( cdouble dt, cint bx, cint by, cint bz )
{
	Diffusion( *dev_den0, *dev_den, DIFFUSION, bx, by, bz );
	std::swap( *dev_den0, *dev_den );
	Advection( *dev_den, *dev_den0, *dev_u, *dev_v, *dev_w, dt, bx, by, bz );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};


void FluidSimProc::Jacobi
	( double *out, cdouble *in, cdouble diff, cdouble divisor, cint bx, cint by, cint bz )
{
	for ( int k=0; k<20; k++)
		kernelJacobi __device_func__ ( out, in, bx, by, bz, diff, divisor );
};

void FluidSimProc::Advection
	( double *out, cdouble *in, cdouble *u, cdouble *v, cdouble *w, cdouble dt, cint bx, cint by, cint bz )
{	
	kernelAdvection __device_func__ ( out, in, bx, by, bz, dt, u, v, w );
};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff, cint bx, cint by, cint bz )
{
	double rate = diff * bx * by * bz;
	Jacobi( out, in, rate, 1+6*rate, bx, by, bz );
};

void FluidSimProc::Projection
	( double *u, double *v, double *w, double *div, double *p, cint bx, cint by, cint bz )
{
	// the velocity gradient
	kernelGradient __device_func__ ( div, p, bx, by, bz, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f, bx, by, bz);

	// now subtract this gradient from our current velocity field
	kernelSubtract __device_func__ ( u, v, w, p, bx, by, bz );
};