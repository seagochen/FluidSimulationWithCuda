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

	kernelSetBound __device_func__ ( gGrids_obstacle, gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	kernelLoadBullet __device_func__
		( gBullets_obstacle, gGrids_obstacle,
		gBULLET_X, gBULLET_Y, gBULLET_Z, gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};


void FluidSimProc::GenerVolumeImg( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		gGRIDS_X / 2, gGRIDS_Y / 2, gGRIDS_X, gGRIDS_Y, gGRIDS_Z );

	kernelExitBullet __device_func__ ( gGrids_density, gBullets_density, 
		gGRIDS_X, gGRIDS_Y, gGRIDS_Z,
		gBULLET_X, gBULLET_Y, gBULLET_Z );


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S,
		TILE_X, TILE_Y, BIG_X, BIG_Y, BIG_Z );

	kernelUpScalingInterpolation __device_func__ ( Big_density, gGrids_density,
		gGRIDS_X, gGRIDS_Y, gGRIDS_Z,
		BIG_X, BIG_Y, BIG_Z,
		4, 4, 4 );

	kernelPickData __device_func__ ( m_ptrDeviceVisual, Big_density, VOLUME_X, VOLUME_Y, VOLUME_Z );


	if ( m_scHelper.GetCUDALastError( "call member function GenerVolumeImg failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};


void FluidSimProc::SolveGlobalFlux()
{
	dev_den  = gBullets_density;
	dev_u    = gBullets_velocity_u;
	dev_v    = gBullets_velocity_v;
	dev_w    = gBullets_velocity_w;
	dev_p    = gBullets_pressure;
	dev_div  = gBullets_divergence;
	dev_obs  = gBullets_obstacle;
	dev_den0 = gBullets_density0;
	dev_u0   = gBullets_velocity_u0;
	dev_v0   = gBullets_velocity_v0;
	dev_w0   = gBullets_velocity_w0;

	//TODO

	if ( m_scHelper.GetCUDALastError( "call member function SolveGlobalFlux failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};


void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( not fluid->run ) return;

	SolveGlobalFlux();

	SolveNavierStokesEquation( DELTATIME, true, true, true, gBULLET_X, gBULLET_Y, gBULLET_Z );
	
	GenerVolumeImg();

	RefreshStatus( fluid );
};


void FluidSimProc::SolveNavierStokesEquation
	( cdouble dt, bool add, bool vel, bool den, cint bx, cint by, cint bz )
{
	if (add) SourceSolver( dt, bx, by, bz );
	if (vel) VelocitySolver( dt, bx, by, bz );
	if (den) DensitySolver( dt, bx, by, bz );
};

//TODO