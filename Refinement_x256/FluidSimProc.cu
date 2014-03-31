/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 25, 2014
* <File Name>     FluidSimProc.cu
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

FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
	{
		m_scHelper.GetCUDALastError( "cannot set device", __FILE__, __LINE__ );
		exit(1);
	}

	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	AllocateResource();
	
	/* clear buffer */
	ClearBuffers();

	/* create boundary condition */
	InitBoundary();

	/* finally, print message */
	printf( "fluid simulation ready...\n" );
};


void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	fluid->fps.dwCurrentTime = 0;
	fluid->fps.dwElapsedTime = 0;
	fluid->fps.dwFrames = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS = 0;

	srand(time(NULL));

	m_szTitle = APP_TITLE;
};


void FluidSimProc::AllocateResource( void )
{
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectgGrids,
		gGRIDS_X * gGRIDS_Y * gGRIDS_Z * sizeof(double), 5 ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectgBullets,
		gBULLET_X * gBULLET_Y * gBULLET_Z * sizeof(double), 11 ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectBigBuffers, 
		BIG_X * BIG_Y * BIG_Z * sizeof(double), 5 ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectsGrids, 
		sGRIDS_X * sGRIDS_Y * sGRIDS_Z * sizeof(double), 5 ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectsBullets, 
		sBULLET_X * sBULLET_Y * sBULLET_Z * sizeof(double), 11 ) ) goto Error;

	m_scHelper.CreateDeviceBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR),
		1, &m_ptrDeviceVisual );
	m_scHelper.CreateHostBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR),
		1, &m_ptrHostVisual );

	goto Success;

Error:
		cout << "create computation buffers device failed" << endl;
		FreeResource();
		exit(1);

Success:
		;
};


void FluidSimProc::FreeResource( void )
{
	int i;

	for ( i = 0; i < m_vectgGrids.size(); i++ ) m_scHelper.FreeDeviceBuffers( 1, & m_vectgGrids[i] );
	for ( i = 0; i < m_vectgBullets.size(); i++ ) m_scHelper.FreeDeviceBuffers( 1, &m_vectgBullets[i] );
	for ( i = 0; i < m_vectBigBuffers.size(); i++ ) m_scHelper.FreeDeviceBuffers( 1, &m_vectBigBuffers[i] );
	for ( i = 0; i < m_vectsGrids.size(); i++ ) m_scHelper.FreeDeviceBuffers( 1, &m_vectsGrids[i] );
	for ( i = 0; i < m_vectsBullets.size(); i++ ) m_scHelper.FreeDeviceBuffers( 1, &m_vectsBullets[i] );

	m_scHelper.FreeDeviceBuffers( 1, &m_ptrDeviceVisual );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostVisual );
}


void FluidSimProc::RefreshStatus( FLUIDSPARAM *fluid )
{
	/* waiting for all kernels end */
	if ( cudaThreadSynchronize() not_eq cudaSuccess )
	{
		printf( "cudaThreadSynchronize failed\n" );
		FreeResource();
		exit( 1 );
	}

	/* counting FPS */
	fluid->fps.dwFrames ++;
	fluid->fps.dwCurrentTime = GetTickCount();
	fluid->fps.dwElapsedTime = fluid->fps.dwCurrentTime - fluid->fps.dwLastUpdateTime;

	/* 1 second */
	if ( fluid->fps.dwElapsedTime >= 1000 )
	{
		fluid->fps.uFPS     = fluid->fps.dwFrames * 1000 / fluid->fps.dwElapsedTime;
		fluid->fps.dwFrames = 0;
		fluid->fps.dwLastUpdateTime = fluid->fps.dwCurrentTime;
	}

	/* updating image */
	if ( cudaMemcpy( m_ptrHostVisual, m_ptrDeviceVisual, 
		VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_scHelper.GetCUDALastError( "host function: cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	fluid->volume.ptrData = m_ptrHostVisual;
};


void FluidSimProc::ClearBuffers( void )
{
	int i;

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		gGRIDS_X / 2, gGRIDS_Y / 2, gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	for ( i = 0; i < m_vectgGrids.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectgGrids[i], gGRIDS_X, gGRIDS_Y, gGRIDS_Z );


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 
		gBULLET_X / 2, gBULLET_Y / 3, gBULLET_X, gBULLET_Y, gBULLET_Z );

	for ( i = 0; i < m_vectgBullets.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectgBullets[i], gBULLET_X, gBULLET_Y, gBULLET_Z );


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		sGRIDS_X / 4, sGRIDS_Y / 4, sGRIDS_X, sGRIDS_Y, sGRIDS_Z );

	for ( i = 0; i < m_vectsGrids.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectsGrids[i], sGRIDS_X, sGRIDS_Y, sGRIDS_Z );


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		sBULLET_X / 5, sBULLET_Y / 5, sBULLET_X, sBULLET_Y, sBULLET_Z );

	for ( i = 0; i < m_vectsBullets.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectsBullets[i], sBULLET_X, sBULLET_Y, sBULLET_Z );


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		BIG_X / 8, BIG_Y / 8, BIG_X, BIG_Y, BIG_Z );
	for ( i = 0; i < m_vectBigBuffers.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectBigBuffers[i], BIG_X, BIG_Y, BIG_Z );

	kernelZeroBuffers __device_func__ ( m_ptrDeviceVisual, VOLUME_X, VOLUME_Y, VOLUME_Z );


	if ( m_scHelper.GetCUDALastError( "call member function ClearBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
}


void FluidSimProc::InitBoundary( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		TILE_X, TILE_Y, gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	kernelSetBound __device_func__ ( m_vectgGrids[DEV_OBSTACLE], gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	kernelLoadBullet __device_func__
		( m_vectgBullets[DEV_OBSTACLE], m_vectgGrids[DEV_OBSTACLE],
		gBULLET_X, gBULLET_Y, gBULLET_Z, gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};


void FluidSimProc::GenerVolumeImg( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 256, 256, 256 );

//	kernelPickData __device_func__ ( m_ptrDeviceVisual, m_vectBigBuffers[DEV_DENSITY],
//		VOLUME_X, VOLUME_Y, VOLUME_Z );


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 128, 128, 128 );

	kernelExitBullet __device_func__ ( m_vectsGrids[DEV_DENSITY], m_vectsBullets[DEV_DENSITY],
		128, 128, 128, 130, 130, 130 );

	kernelPickData __device_func__ ( m_ptrDeviceVisual, m_vectsGrids[DEV_DENSITY], 
		VOLUME_X, VOLUME_Y, VOLUME_Z,
		128, 128, 128,
		1, 1, 1 );

	if ( m_scHelper.GetCUDALastError( "call member function GenerVolumeImg failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

#if 0
#define dev_den m_vectgBullets[DEV_DENSITY]
#define	dev_u   m_vectgBullets[DEV_VELOCITY_U]
#define	dev_v   m_vectgBullets[DEV_VELOCITY_V]
#define	dev_w   m_vectgBullets[DEV_VELOCITY_W]
#define	dev_p   m_vectgBullets[DEV_PRESSURE]
#define	dev_div m_vectgBullets[DEV_DIVERGENCE]
#define	dev_obs m_vectgBullets[DEV_OBSTACLE]
#define	dev_den0 m_vectgBullets[DEV_DENSITY0]
#define	dev_u0   m_vectgBullets[DEV_VELOCITY_U0]
#define	dev_v0   m_vectgBullets[DEV_VELOCITY_V0]
#define	dev_w0   m_vectgBullets[DEV_VELOCITY_W0]
#endif

void FluidSimProc::SolveGlobalFlux( void )
{
#if 1
	dev_den  = &m_vectgBullets[DEV_DENSITY];
	dev_u    = &m_vectgBullets[DEV_VELOCITY_U];
	dev_v    = &m_vectgBullets[DEV_VELOCITY_V];
	dev_w    = &m_vectgBullets[DEV_VELOCITY_W];
	dev_p    = &m_vectgBullets[DEV_PRESSURE];
	dev_div  = &m_vectgBullets[DEV_DIVERGENCE];
	dev_obs  = &m_vectgBullets[DEV_OBSTACLE];
	dev_den0 = &m_vectgBullets[DEV_DENSITY0];
	dev_u0   = &m_vectgBullets[DEV_VELOCITY_U0];
	dev_v0   = &m_vectgBullets[DEV_VELOCITY_V0];
	dev_w0   = &m_vectgBullets[DEV_VELOCITY_W0];


	if ( m_scHelper.GetCUDALastError( "call member function SolveGlobalFlux failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
#endif
};


void FluidSimProc::RefinementFlux( void )
{

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 64, 64, 64 );

	kernelExitBullet __device_func__ ( m_vectgGrids[DEV_DENSITY], m_vectgBullets[DEV_DENSITY], 
		64, 64, 64,  66, 66, 66 );

	kernelExitBullet __device_func__ ( m_vectgGrids[DEV_VELOCITY_U], m_vectgBullets[DEV_VELOCITY_U], 
		64, 64, 64,  66, 66, 66 );
	
	kernelExitBullet __device_func__ ( m_vectgGrids[DEV_VELOCITY_V], m_vectgBullets[DEV_VELOCITY_V], 
		64, 64, 64,  66, 66, 66 );

	kernelExitBullet __device_func__ ( m_vectgGrids[DEV_VELOCITY_W], m_vectgBullets[DEV_VELOCITY_W], 
		64, 64, 64,  66, 66, 66 );


	if ( m_scHelper.GetCUDALastError( "call member function GenerVolumeImg failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, BIG_X, BIG_Y, BIG_Z );

	kernelUpScalingInterpolation __device_func__
		( m_vectBigBuffers[DEV_DENSITY], m_vectgGrids[DEV_DENSITY],
		64, 64, 64, 256, 256, 256, 4, 4, 4 );

	kernelUpScalingInterpolation __device_func__
		( m_vectBigBuffers[DEV_VELOCITY_U], m_vectgGrids[DEV_VELOCITY_U],
		64, 64, 64, 256, 256, 256, 4, 4, 4 );

	kernelUpScalingInterpolation __device_func__
		( m_vectBigBuffers[DEV_VELOCITY_V], m_vectgGrids[DEV_VELOCITY_V],
		64, 64, 64, 256, 256, 256, 4, 4, 4 );

	kernelUpScalingInterpolation __device_func__
		( m_vectBigBuffers[DEV_VELOCITY_W], m_vectgGrids[DEV_VELOCITY_W],
		64, 64, 64, 256, 256, 256, 4, 4, 4 );

	kernelUpScalingInterpolation __device_func__
		( m_vectBigBuffers[DEV_OBSTACLE], m_vectgGrids[DEV_OBSTACLE],
		64, 64, 64, 256, 256, 256, 4, 4, 4 );

	if ( m_scHelper.GetCUDALastError( "call member function GenerVolumeImg failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}


	m_scHelper.DeviceParamDim 
		( &gridDim, &blockDim, THREADS_S, 26, 26, sBULLET_X, sBULLET_Y, sBULLET_Z );

	kernelFillBullet __device_func__ ( m_vectsBullets[DEV_DENSITY], m_vectBigBuffers[DEV_DENSITY],
		256, 256, 256, 
		130, 130, 130,
		128, 128, 128,
		1, 1, 1 );

	kernelFillBullet __device_func__ ( m_vectsBullets[DEV_VELOCITY_U], m_vectBigBuffers[DEV_VELOCITY_U],
		256, 256, 256, 
		130, 130, 130,
		128, 128, 128,
		1, 1, 1 );

	kernelFillBullet __device_func__ ( m_vectsBullets[DEV_VELOCITY_V], m_vectBigBuffers[DEV_VELOCITY_V],
		256, 256, 256, 
		130, 130, 130,
		128, 128, 128,
		1, 1, 1 );

	kernelFillBullet __device_func__ ( m_vectsBullets[DEV_VELOCITY_W], m_vectBigBuffers[DEV_VELOCITY_W],
		256, 256, 256, 
		130, 130, 130,
		128, 128, 128,
		1, 1, 1 );

	kernelFillBullet __device_func__ ( m_vectsBullets[DEV_OBSTACLE], m_vectBigBuffers[DEV_OBSTACLE],
		256, 256, 256, 
		130, 130, 130,
		128, 128, 128,
		1, 1, 1 );

	if ( m_scHelper.GetCUDALastError( "call member function GenerVolumeImg failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	dev_den  = &m_vectsBullets[DEV_DENSITY];
	dev_u    = &m_vectsBullets[DEV_VELOCITY_U];
	dev_v    = &m_vectsBullets[DEV_VELOCITY_V];
	dev_w    = &m_vectsBullets[DEV_VELOCITY_W];
	dev_p    = &m_vectsBullets[DEV_PRESSURE];
	dev_div  = &m_vectsBullets[DEV_DIVERGENCE];
	dev_obs  = &m_vectsBullets[DEV_OBSTACLE];
	dev_den0 = &m_vectsBullets[DEV_DENSITY0];
	dev_u0   = &m_vectsBullets[DEV_VELOCITY_U0];
	dev_v0   = &m_vectsBullets[DEV_VELOCITY_V0];
	dev_w0   = &m_vectsBullets[DEV_VELOCITY_W0];

};


void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( not fluid->run ) return;

	SolveGlobalFlux();

	SolveNavierStokesEquation( DELTATIME, true, true, true,
		32, 32, gGRIDS_X, gGRIDS_Y, gGRIDS_Z, gBULLET_X, gBULLET_Y, gBULLET_Z );

	RefinementFlux();
	
	SolveNavierStokesEquation( DELTATIME, false, true, true,
		32, 32, sGRIDS_X, sGRIDS_Y, sGRIDS_Z, sBULLET_X, sBULLET_Y, sBULLET_Z );

	GenerVolumeImg();

	RefreshStatus( fluid );
};


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