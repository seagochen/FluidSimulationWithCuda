/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 25, 2014
* <File Name>     FluidSimProc.cu
*/

#include <iostream>
#include <utility>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "ExtFunctions.h"
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

	/* initialize boundary */
	InitBound();

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

	m_szTitle = APP_TITLE;
};

void FluidSimProc::InitBound( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 22, 22, BULLET_X, BULLET_Y, BULLET_Z );
	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectCompBufs[i], BULLET_X, BULLET_Y, BULLET_Z );
	}

	if ( m_scHelper.GetCUDALastError( "call member function InitBound failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
}

void FluidSimProc::AllocateResource( void )
{
	if ( not CreateCompNodesForHost(
		&m_vectHostDens, 
		&m_vectHostVelU,
		&m_vectHostVelV,
		&m_vectHostVelW, 
		&m_vectHostObst,
		&m_scHelper, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double),
		NODES_X * NODES_Y * NODES_Z ))
	{
		cout << "create computation nodes for host failed" << endl;
		FreeResource();
		exit(1);
	}

	if ( not CreateCompNodesForDevice(
		&m_vectGPUDens, 
		&m_vectGPUVelU,
		&m_vectGPUVelV,
		&m_vectGPUVelW,
		&m_vectGPUObst,
		&m_scHelper,
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double),
		NODES_X * NODES_Y * NODES_Z ))
	{
		cout << "create computation nodes layer 0 for device failed" << endl;
		FreeResource();
		exit(1);
	}

	if ( not CreateCompNodesForDevice(
		&m_vectNewDens, 
		&m_vectNewVelU,
		&m_vectNewVelV,
		&m_vectNewVelW,
		&m_vectNewObst,
		&m_scHelper,
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double),
		NODES_X * NODES_Y * NODES_Z ))
	{
		cout << "create computation nodes layer 1 for device failed" << endl;
		FreeResource();
		exit(1);
	}

	if ( not CreateCompNodesForDevice( &m_vectCompBufs, &m_scHelper, 
		BULLET_X * BULLET_Y * BULLET_Z * sizeof(double),
		BULLET_S ) )
	{
		cout << "create computation buffers for device failed" << endl;
		FreeResource();
		exit(1);
	}

	m_scHelper.CreateDeviceBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrDeviceVisual );
	m_scHelper.CreateHostBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrHostVisual );

	if ( m_scHelper.GetCUDALastError( "call member function AllocateResource failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::FreeResource( void )
{
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		m_scHelper.FreeDeviceBuffers
			( 5, &m_vectGPUDens[i], &m_vectGPUVelU[i], &m_vectGPUVelV[i], &m_vectGPUVelW[i], &m_vectGPUObst[i] );
		m_scHelper.FreeDeviceBuffers
			( 5, &m_vectNewDens[i], &m_vectNewVelU[i], &m_vectNewVelV[i], &m_vectNewVelW[i], &m_vectNewObst[i] );
		m_scHelper.FreeHostBuffers
			( 5, &m_vectHostDens[i], &m_vectHostVelU[i], &m_vectHostVelV[i], &m_vectHostVelW[i], &m_vectHostObst[i] );
	}

	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
	{
		m_scHelper.FreeDeviceBuffers( 1, &m_vectCompBufs[i] );
	}

	m_scHelper.FreeDeviceBuffers( 1, &m_ptrDeviceVisual );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostVisual );

	if ( m_scHelper.GetCUDALastError( "call member function FreeResource failed",
		__FILE__, __LINE__ ) ) exit(1);
};

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

	if ( m_scHelper.GetCUDALastError( "call member function RefreshStatus failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;
	
	Dim3ParamDim();

	kernelLoadBullet __device_func__ ( dev_den, m_vectGPUDens[0], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_u, m_vectGPUVelU[0], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_v, m_vectGPUVelV[0], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_w, m_vectGPUVelW[0], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_obs, m_vectGPUObst[0], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );

	SolveNavierStokesEquation( DELTATIME, true );

	kernelExitBullet __device_func__ ( m_vectGPUDens[0], dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectGPUVelU[0], dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectGPUVelV[0], dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectGPUVelW[0], dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	kernelPickData __device_func__ ( m_ptrDeviceVisual, m_vectGPUDens[0], 0, 0, 0, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function FluidSimSolver failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	RefreshStatus( fluid );
};

void FluidSimProc::ClearBuffers( void )
{
	Dim3ParamDim();
	
	for ( int i = 0; i < m_vectCompBufs.size(); i++ ) _zero( m_vectCompBufs[i] );
		 	 

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		_zero( m_vectGPUDens[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelU[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelV[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelW[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUObst[i] ); _zero( m_vectNewDens[i] );
	}

	if ( m_scHelper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};