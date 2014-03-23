/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 19, 2014
* <File Name>     FluidSimProc.cu
*/

#include <time.h>
#include <iostream>
#include <utility>
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "FrameworkDynamic.h"
#include "Kernels.h"
#include "SimulationParam.h"

using std::cout;
using std::endl;

using namespace sge;


FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
	{
		printf( "cudaSetDevice failed!\n" );
		exit(1);
	}

	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	AllocateResource();

	/* build order */
	CreateTopology();
	
	/* clear buffer */
	ZeroBuffers();

	/* set boundary */
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

	m_nNodeSize = GRIDS_X * GRIDS_Y * GRIDS_Z;
	m_nVolumSize = VOLUME_X * VOLUME_Y * VOLUME_Z;
	m_nBulletSize = BULLET_X * BULLET_Y * BULLET_Z;

	m_nNodeNum = NODES_X * NODES_Y * NODES_Z;
	m_nBulletNum = BULLET_S;

	m_nDensIncrease = m_nDensDecrease = 0;
	m_ptrSimNode = nullptr;
	m_szTitle = "Xcalibur OTL 3.10.00 ------------ FPS: %d ";

	cout << "fluid simulation parameter initialized" << endl;
};

void FluidSimProc::AllocateResource( void )
{
	if ( not CreateCompNodes() ) { FreeResource(); exit(1); }
	if ( not CreateDeviceBuffers() ) { FreeResource(); exit(1); }
	if ( not CreateVisualBuffers() ) { FreeResource(); exit(1); }
	
	if ( m_scHelper.GetCUDALastError( "memory allocation failed, check the code", __FILE__, __LINE__ ) )
	{ FreeResource(); exit(1); }

	cout << "resource for fluid simulation created" <<endl;
};

bool FluidSimProc::CreateCompNodes( void )
{
	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( m_scHelper.CreateHostBuffers( m_nNodeSize * sizeof(double), 1, &ptrDens )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateHostBuffers( m_nNodeSize * sizeof(double), 1, &ptrU ) 
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateHostBuffers( m_nNodeSize * sizeof(double), 1, &ptrV )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateHostBuffers( m_nNodeSize * sizeof(double), 1, &ptrW )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateHostBuffers( m_nNodeSize * sizeof(double), 1, &ptrObs ) 
			not_eq SG_RUNTIME_OK ) return false;

		m_vectHostDens.push_back( ptrDens );
		m_vectHostVelU.push_back( ptrU );
		m_vectHostVelV.push_back( ptrV );
		m_vectHostVelW.push_back( ptrW );
		m_vectHostObst.push_back( ptrObs );
	}

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrDens )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrU ) 
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrV )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrW )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrObs ) 
			not_eq SG_RUNTIME_OK ) return false;

		m_vectGPUDens.push_back( ptrDens );
		m_vectGPUVelU.push_back( ptrU );
		m_vectGPUVelV.push_back( ptrV );
		m_vectGPUVelW.push_back( ptrW );
		m_vectGPUObst.push_back( ptrObs );
	}

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW;

		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrDens )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrU ) 
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrV )
			not_eq SG_RUNTIME_OK ) return false;
		if ( m_scHelper.CreateDeviceBuffers( m_nNodeSize * sizeof(double), 1, &ptrW )
			not_eq SG_RUNTIME_OK ) return false;

		m_vectNewDens.push_back( ptrDens );
		m_vectNewVelU.push_back( ptrU );
		m_vectNewVelV.push_back( ptrV );
		m_vectNewVelW.push_back( ptrW );
	}

	return true;
};

bool FluidSimProc::CreateDeviceBuffers( void )
{
	for ( int i = 0; i < m_nBulletNum; i++ )
	{
		double *ptr;

		if ( m_scHelper.CreateDeviceBuffers( m_nBulletSize * sizeof(double), 1, &ptr )
			not_eq SG_RUNTIME_OK )
			return false;

		m_vectGPUBuffers.push_back(ptr);
	}

	if ( m_scHelper.CreateDeviceBuffers( TPBUFFER_S * sizeof(double), 1, &m_ptrGPUShare )
		not_eq SG_RUNTIME_OK ) return false;
	if ( m_scHelper.CreateHostBuffers( TPBUFFER_S * sizeof(double), 1, &m_ptrHostShare )
		not_eq SG_RUNTIME_OK ) return false;


	return true;
};

bool FluidSimProc::CreateVisualBuffers( void )
{
	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		node->active   = false;

		m_vectLink.push_back( node );
	}
	
	if ( m_scHelper.CreateDeviceBuffers( m_nVolumSize * sizeof(uchar), 1, &m_ptrGPUVisual ) not_eq SG_RUNTIME_OK ) return false;
	if ( m_scHelper.CreateHostBuffers( m_nVolumSize * sizeof(uchar), 1, &m_ptrHostVisual ) not_eq SG_RUNTIME_OK ) return false;

	return true;
};

void FluidSimProc::CreateTopology( void )
{
	for ( int k = 0; k < NODES_X; k++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int i = 0; i < NODES_X; i++ )
			{
				/* left */
				if ( i >= 1 )
					m_vectLink[ IX( i,j,k,NODES_X )]->ptrLeft  = m_vectLink[ IX( i-1,j,k,NODES_X ) ];
				/* right */
				if ( i <= NODES_X - 2 )
					m_vectLink[ IX( i,j,k,NODES_X )]->ptrRight = m_vectLink[ IX( i+1,j,k,NODES_X ) ];
				/* down */
				if ( j >= 1 )
					m_vectLink[ IX( i,j,k,NODES_X )]->ptrDown  = m_vectLink[ IX( i,j-1,k,NODES_X ) ];
				/* up */
				if ( j <= NODES_X - 2 )
					m_vectLink[ IX( i,j,k,NODES_X )]->ptrUp    = m_vectLink[ IX( i,j+1,k,NODES_X ) ];
				/* back */
				if ( k >= 1 )
					m_vectLink[ IX( i,j,k,NODES_X )]->ptrBack  = m_vectLink[ IX( i,j,k-1,NODES_X ) ];
				/* front */
				if ( k <= NODES_X - 2 )
					m_vectLink[ IX( i,j,k,NODES_X )]->ptrFront = m_vectLink[ IX( i,j,k+1,NODES_X ) ];
			}
		}
	}

	cout << "nodes' topology created" << endl;
};

void FluidSimProc::ZeroBuffers( void )
{
	ClearCompNodes();
	ClearDeviceBuffers();
	ClearVisualBuffers();
	
	if ( m_scHelper.GetCUDALastError( "call member function zerobuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	cout << "buffers on both device and host cleared." << endl;
};

void FluidSimProc::InitBoundary( void )
{
	int const halfx = GRIDS_X / 2;
	int const halfz = GRIDS_Z / 2;

	for ( int k = 0; k < GRIDS_Z; k++ )
	{
		for ( int j = 0; j < GRIDS_Y; j++ )
		{
			for ( int i = 0; i < GRIDS_X; i++ )
			{
				if ( j < 2 and i < halfx + 3 and i >= halfx - 3 and 
					k < halfz + 3 and k >= halfz - 3 )
				m_vectHostObst[0][IX(i,j,k,GRIDS_X,GRIDS_Y,GRIDS_Z)] = MACRO_BOUNDARY_SOURCE;
			}
		}
	}

	cudaMemcpy( m_vectGPUObst[0], m_vectHostObst[0], sizeof(double) * m_nNodeSize, cudaMemcpyHostToDevice );

	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	cout << "boundary for fluid simulation is ok!" << endl;
};

void FluidSimProc::FreeResource ( void )
{
	ReleaseCompNodes();
	ReleaseDeviceBuffers();
	ReleaseVisualBuffers();
}

void FluidSimProc::ReleaseCompNodes( void )
{
	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		m_scHelper.FreeDeviceBuffers( 2, &m_vectGPUDens[i], &m_vectNewDens[i] );
		m_scHelper.FreeDeviceBuffers( 2, &m_vectGPUVelU[i], &m_vectNewVelU[i] );
		m_scHelper.FreeDeviceBuffers( 2, &m_vectGPUVelV[i], &m_vectNewVelV[i] );
		m_scHelper.FreeDeviceBuffers( 2, &m_vectGPUVelW[i], &m_vectNewVelW[i] );
		m_scHelper.FreeDeviceBuffers( 1, &m_vectGPUObst[i] );

		m_scHelper.FreeHostBuffers( 1, &m_vectHostDens[i] );
		m_scHelper.FreeHostBuffers( 1, &m_vectHostVelU[i] );
		m_scHelper.FreeHostBuffers( 1, &m_vectHostVelV[i] );
		m_scHelper.FreeHostBuffers( 1, &m_vectHostVelW[i] );
		m_scHelper.FreeHostBuffers( 1, &m_vectHostObst[i] );
	}

	cout << "computation nodes released" << endl;
};

void FluidSimProc::ReleaseDeviceBuffers( void )
{
	for ( int i = 0; i < m_vectGPUBuffers.size(); i++ )
	{
		m_scHelper.FreeDeviceBuffers( 1, &m_vectGPUBuffers[i] );
	}

	m_scHelper.FreeDeviceBuffers( 1, &m_ptrGPUShare );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostShare );

	cout << "device buffers released" << endl;
};

void FluidSimProc::ReleaseVisualBuffers( void )
{
	m_scHelper.FreeDeviceBuffers( 1, &m_ptrGPUVisual );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostVisual );

	cout << "visual buffers released" << endl;
};

void FluidSimProc::SaveCurFluidSimStatus( void )
{
	cout << "saving current fluid simulation status" << endl;

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		cudaMemcpy( m_vectHostDens[i], m_vectNewDens[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelU[i], m_vectNewVelU[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelV[i], m_vectNewVelV[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelW[i], m_vectNewVelW[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
	}

	if ( m_scHelper.GetCUDALastError("call member function SaveCurFluidSimStatus failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	cout << "current fluid simulation status saved" << endl;
};

void FluidSimProc::LoadPreFluidSimStatus( void )
{
	cout << "load previous fluid simulation status" << endl;

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		cudaMemcpy( m_vectGPUDens[i], m_vectHostDens[i], sizeof(double) * m_nNodeSize, cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectGPUVelU[i], m_vectHostVelU[i], sizeof(double) * m_nNodeSize, cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectGPUVelV[i], m_vectHostVelV[i], sizeof(double) * m_nNodeSize, cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectGPUVelW[i], m_vectHostVelW[i], sizeof(double) * m_nNodeSize, cudaMemcpyHostToDevice );
	}

	if ( m_scHelper.GetCUDALastError("call member function LoadPreFluidSimStatus failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	cout << "previous fluid simulation status loaded" << endl;
};

void FluidSimProc::SwapComNodes( void )
{
	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		std::swap( m_vectGPUDens[i], m_vectNewDens[i] );
		std::swap( m_vectGPUVelU[i], m_vectNewVelU[i] );
		std::swap( m_vectGPUVelV[i], m_vectNewVelV[i] );
		std::swap( m_vectGPUVelW[i], m_vectNewVelW[i] );
	}
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( not fluid->run ) return;

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		PushCompNode( i );
		
		m_scSolver.SolveSource( dev_den, dev_u, dev_v, dev_w, dev_obs, &m_nDensIncrease, &m_nDensDecrease, DELTATIME );
		m_scSolver.SolveVelocity( dev_u0, dev_v0, dev_w0, dev_u, dev_v, dev_w, dev_div, dev_p, DELTATIME );
		m_scSolver.SolveDensity( dev_den0, dev_den, dev_u, dev_v, dev_w, DELTATIME );

		if ( m_scHelper.GetCUDALastError( "call member function FluidSimSolver failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit(1);
		}

		PopCompNode( i );
	}

	SwapComNodes();
	GenVolumeImage( fluid );
};




#define __device_func__ <<<gridDim, blockDim>>>

void FluidSimProc::ClearCompNodes( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectGPUDens[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUVelU[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUVelV[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUVelW[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUObst[i], GRIDS_X, GRIDS_Y, GRIDS_Z );

		kernelZeroBuffers __device_func__ ( m_vectNewDens[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectNewVelU[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectNewVelV[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectNewVelW[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( m_scHelper.GetCUDALastError( "call member function ClearCompNodes failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		cudaMemcpy( m_vectHostDens[i], m_vectNewDens[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelU[i], m_vectNewVelU[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelV[i], m_vectNewVelV[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelW[i], m_vectNewVelW[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostObst[i], m_vectGPUObst[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
	}

	if ( m_scHelper.GetCUDALastError( "call member function ClearCompNodes failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::ClearDeviceBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 22, 22, BULLET_X, BULLET_Y, BULLET_Z );

	for ( int i = 0; i < m_vectGPUBuffers.size(); i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectGPUBuffers[i], BULLET_X, BULLET_Y, BULLET_Z );
	}

	if ( m_scHelper.GetCUDALastError( "call member function ClearDeviceBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TPBUFFER_S );

	kernelZeroBuffers __device_func__ ( m_ptrGPUShare, TPBUFFER_S );

	if ( m_scHelper.GetCUDALastError( "call member function ClearDeviceBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	cudaMemcpy( m_ptrHostShare, m_ptrGPUShare, sizeof(double) * TPBUFFER_S, cudaMemcpyDeviceToHost );

	if ( m_scHelper.GetCUDALastError( "call member function ClearDeviceBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::ClearVisualBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, VOLUME_X, VOLUME_Y, VOLUME_Z );

	kernelZeroBuffers __device_func__ ( m_ptrGPUVisual, VOLUME_X, VOLUME_Y, VOLUME_Z );
	cudaMemcpy( m_ptrHostVisual, m_ptrGPUVisual, sizeof(uchar) * m_nVolumSize, cudaMemcpyDeviceToHost );

	if ( m_scHelper.GetCUDALastError( "call member function ClearVisualBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::PushCompNode( int id )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelLoadBullet __device_func__ ( dev_den, m_vectGPUDens[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_u, m_vectGPUVelU[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_v, m_vectGPUVelV[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_w, m_vectGPUVelW[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_obs, m_vectGPUObst[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function PushCompNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::PopCompNode( int id )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelExitBullet __device_func__ ( m_vectNewDens[id], dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelU[id], dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelV[id], dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelW[id], dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	if ( m_scHelper.GetCUDALastError( "call member function PopCompNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::GenVolumeImage( FLUIDSPARAM *fluid )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k = 0; k < NODES_Z; k++ )
	{
		for ( int j = 0; j < NODES_Y; j++ )
		{
			for ( int i = 0; i < NODES_X; i++ )
			{
				kernelDensToVolume __device_func__
					( m_ptrGPUVisual, m_vectGPUDens[IX(i,j,k,NODES_X,NODES_Y,NODES_Z)], i, j, k,
					GRIDS_X, GRIDS_Y, GRIDS_Z, VOLUME_X, VOLUME_Y, VOLUME_Z );
			}
		}
	}

	if ( cudaMemcpy( m_ptrHostVisual, m_ptrGPUVisual, sizeof(uchar) * m_nVolumSize, 
		cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_scHelper.GetCUDALastError( "call member function GenVolumeImage failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	
	fluid->volume.ptrData = m_ptrHostVisual;
};