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

bool CreateCompNodesForDevice
	( vector<double*> *vectDens, vector<double*> *vectVelU, vector<double*> *vectVelV, 
	vector<double*> *vectVelW, vector<double*> *vectObst, 
	FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateDeviceBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectDens->push_back( ptrD );
		vectVelU->push_back( ptrU );
		vectVelV->push_back( ptrV );
		vectVelW->push_back( ptrW );
		vectObst->push_back( ptrO );
	}

	return true;
};

bool CreateCompNodesForDevice( vector<double*> *vectBuf, FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateDeviceBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptrD );
	}

	return true;
};

bool CreateCompNodesForHost
	( vector<double*> *vectDens, vector<double*> *vectVelU, vector<double*> *vectVelV, vector<double*> *vectVelW,
	vector<double*> *vectObst, FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateHostBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectDens->push_back( ptrD );
		vectVelU->push_back( ptrU );
		vectVelV->push_back( ptrV );
		vectVelW->push_back( ptrW );
		vectObst->push_back( ptrO );
	}

	return true;
};




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

	if ( not CreateCompNodesForDevice( &m_vectCompBufs, &m_scHelper, GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 40 ) )
	{
		cout << "create computation buffers for device failed" << endl;
		FreeResource();
		exit(1);
	}

	m_scHelper.CreateDeviceBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrDeviceVisual );
	m_scHelper.CreateHostBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrHostVisual );
};

void FluidSimProc::FreeResource( void )
{
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		m_scHelper.FreeDeviceBuffers( 5, &m_vectGPUDens[i], &m_vectGPUVelU[i], &m_vectGPUVelV[i], &m_vectGPUVelW[i], &m_vectGPUObst[i] );
		m_scHelper.FreeDeviceBuffers( 5, &m_vectNewDens[i], &m_vectNewVelU[i], &m_vectNewVelV[i], &m_vectNewVelW[i], &m_vectNewObst[i] );
		m_scHelper.FreeHostBuffers( 5, &m_vectHostDens[i], &m_vectHostVelU[i], &m_vectHostVelV[i], &m_vectHostVelW[i], &m_vectHostObst[i] );
	}

	m_scHelper.FreeDeviceBuffers( 1, &m_ptrDeviceVisual );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostVisual );
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
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;
	
	DeviceParamDim();

	kernelZeroBuffers __device_func__ ( dev_obs, GRIDS_X * GRIDS_Y * GRIDS_Z );

	SolveNavierStokesEquation( DELTATIME, true );

	kernelPickData __device_func__ ( m_ptrDeviceVisual, dev_den, 0, 0, 0, GRIDS_X, GRIDS_Y, GRIDS_Z );

	RefreshStatus( fluid );
};

void FluidSimProc::ClearBuffers( void )
{
	DeviceParamDim();
	
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