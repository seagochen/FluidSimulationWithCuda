/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 26, 2014
* <File Name>     FluidSimProc.cu
*/

#include <iostream>
#include <utility>
#include <ctime>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

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

	srand(time(NULL));

	m_szTitle = APP_TITLE;
	m_enHierarchy = SG_LARGE_SCALE;
};

void FluidSimProc::AllocateResource( void )
{
	size_t size = GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double);
	size_t node = NODES_X * NODES_Y * NODES_Z;
	
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostDens, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelU, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelV, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelW, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostObst, size, node ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectGPUDens, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectGPUVelU, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectGPUVelV, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectGPUVelW, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectGPUObst, size, node ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectNewDens, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectNewVelU, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectNewVelV, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectNewVelW, size, node ) ) goto Error;

	size = GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double);
	node = TEMPND_S;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectGlobalBufs, size, node ) ) goto Error;

	size = BULLET_X * BULLET_Y * BULLET_Z * sizeof(double);
	node = BULLET_S;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectCompBufs, size, node ) ) goto Error;

	goto Success;


Error:
	cout << "create computation nodes failed" << endl;
		FreeResource();
		exit(1);

Success:

	m_scHelper.CreateDeviceBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrDeviceVisual );
	m_scHelper.CreateHostBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrHostVisual );

	if ( m_scHelper.GetCUDALastError( "call member function AllocateResource failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}


	/* create link message and topology */
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		SimNode *link = new SimNode;
		link->ptrBack = link->ptrFront = link->ptrLeft = link->ptrRight = link->ptrUp = link->ptrDown = nullptr;
		link->updated = link->active = MACRO_FALSE;
		m_link.push_back( link );
	}

	for ( int k = 0; k < NODES_Z; k++ )
	{
		for ( int j = 0; j < NODES_Y; j++ )
		{
			for ( int i = 0; i < NODES_X; i++ )
			{
				/* left */
				if ( i >= 1 )
					m_link[ix(i, j, k, NODES_X, NODES_Y)]->ptrLeft  = m_link[ix( i-1, j, k, NODES_X, NODES_Y)];
				/* right */
				if ( i <= NODES_X - 2 )
					m_link[ix(i, j, k, NODES_X, NODES_Y)]->ptrRight = m_link[ix( i+1, j, k, NODES_X, NODES_Y)];
				/* down */
				if ( j >= 1 )
					m_link[ix(i, j, k, NODES_X, NODES_Y)]->ptrDown  = m_link[ix( i, j-1, k, NODES_X, NODES_Y)];
				/* up */
				if ( j <= NODES_Y - 2 )
					m_link[ix(i, j, k, NODES_X, NODES_Y)]->ptrUp    = m_link[ix( i, j+1, k, NODES_X, NODES_Y)];
				/* back */
				if ( k >= 1 )
					m_link[ix(i, j, k, NODES_X, NODES_Y)]->ptrBack  = m_link[ix( i, j, k-1, NODES_X, NODES_Y)];
				/* front */
				if ( k <= NODES_Z - 2 )
					m_link[ix(i, j, k, NODES_X, NODES_Y)]->ptrFront = m_link[ix( i, j, k+1, NODES_X, NODES_Y)];

				m_link[ix(i, j, k, NODES_X, NODES_Y)]->x = i;
				m_link[ix(i, j, k, NODES_X, NODES_Y)]->y = j;
				m_link[ix(i, j, k, NODES_X, NODES_Y)]->z = k;
			}
		}
	}
};

void FluidSimProc::FreeResource( void )
{
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		m_scHelper.FreeDeviceBuffers( 9,
			&m_vectGPUDens[i], &m_vectNewDens[i],
			&m_vectGPUVelU[i], &m_vectNewVelU[i],
			&m_vectGPUVelV[i], &m_vectNewVelV[i],
			&m_vectGPUVelW[i], &m_vectNewVelW[i],
			&m_vectGPUObst[i] );

		m_scHelper.FreeHostBuffers( 5,
			&m_vectHostDens[i], &m_vectHostVelU[i],
			&m_vectHostVelV[i], &m_vectHostVelW[i], &m_vectHostObst[i] );
	}


	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
		SAFE_DELT_PTR( m_link[i] );


	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectCompBufs[i] );


	for ( int i = 0; i < m_vectGlobalBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectGlobalBufs[i] );


	m_scHelper.FreeDeviceBuffers( 1, &m_ptrDeviceVisual );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostVisual );


	if ( m_scHelper.GetCUDALastError( "call member function FreeResource failed",
		__FILE__, __LINE__ ) ) exit(1);
};

void FluidSimProc::RefreshStatus( FLUIDSPARAM *fluid )
{
	/* swap new & old buffers */
//	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
//	{
//		std::swap( m_vectGPUDens[i], m_vectNewDens[i] );
//		std::swap( m_vectGPUVelU[i], m_vectNewVelU[i] );
//		std::swap( m_vectGPUVelV[i], m_vectNewVelV[i] );
//		std::swap( m_vectGPUVelW[i], m_vectNewVelW[i] );
//	}

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

void FluidSimProc::ClearBuffers( void )
{
	GridsParamDim();
	
	for ( int i = 0; i < m_vectCompBufs.size(); i++ ) _zero( m_vectCompBufs[i] );
		 	 

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		_zero( m_vectGPUDens[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelU[i] ); _zero( m_vectNewVelU[i] );
		_zero( m_vectGPUVelV[i] ); _zero( m_vectNewVelV[i] );
		_zero( m_vectGPUVelW[i] ); _zero( m_vectNewVelW[i] );
	}

	for ( int i = 0; i < m_vectGlobalBufs.size(); i++ )
		_zero( m_vectGlobalBufs[i] );

	if ( m_scHelper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::LoadPreStage( void )
{
	/* clear bullet buffers */
	BulletParamDim();
	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectCompBufs[i], BULLET_X, BULLET_Y, BULLET_Z );

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		cudaMemcpy( m_vectGPUDens[i], m_vectHostDens[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectGPUVelU[i], m_vectHostVelU[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectGPUVelV[i], m_vectHostVelV[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectGPUVelW[i], m_vectHostVelW[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
	}

	if ( m_scHelper.GetCUDALastError( "call member function LoadPreStage failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::SaveCurStage( void )
{
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		cudaMemcpy( m_vectHostDens[i], m_vectNewDens[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelU[i], m_vectNewVelU[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelV[i], m_vectNewVelV[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelW[i], m_vectNewVelW[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
	}

	if ( m_scHelper.GetCUDALastError( "call member function SaveCurStage failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::InitBound( void )
{
	cint halfx = GRIDS_X / 2;
	cint halfz = GRIDS_Z / 2;

	for ( int k = 0; k < GRIDS_Z; k++ )
	{
		for ( int j = 0; j < GRIDS_Y; j++ )
		{
			for ( int i = 0; i < GRIDS_X; i++ )
			{
				for ( int n = 0; n < NODES_X * NODES_Y * NODES_Z; n++ )
					m_vectHostObst[n][ix(i,j,k,GRIDS_X,GRIDS_Y)] = MACRO_BOUNDARY_BLANK;
					
				if ( j < 3 and 
					i >= halfx - 2 and i < halfx + 2 and 
					k >= halfz - 2 and k < halfz + 2 )
					m_vectHostObst[0][ix(i,j,k,GRIDS_X,GRIDS_Y)] = MACRO_BOUNDARY_SOURCE;
			}
		}
	}

	if ( m_enHierarchy eqt SG_LARGE_SCALE )
	{
		for ( int n = 0; n < NODES_X * NODES_Y * NODES_Z; n++ )
		{
			cudaMemcpy( m_vectGPUObst[n], m_vectHostObst[n], 
				sizeof(double) * GRIDS_X * GRIDS_Y * GRIDS_Z, cudaMemcpyHostToDevice );
		}
	}
	elif ( m_enHierarchy eqt SG_HIGH_PRECISION )
	{
		cudaMemcpy( temp_obst, m_vectHostObst[0], 
			sizeof(double) * GRIDS_X * GRIDS_Y * GRIDS_Z, cudaMemcpyHostToDevice );
	}

	if ( m_scHelper.GetCUDALastError( "call member function InitBound failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
}

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	/* solve node */
	SolveNavierStokesEquation( DELTATIME, true, true, true );

	/* swap buffers and output image */
	RefreshStatus( fluid );
};

#if 0
void FluidSimProc::SetCurrentNode( int i, int j, int k )
{
	GridsParamDim();

	int ind = ix( i, j, k, NODES_X, NODES_Y );

	/* load current node to bullet */
	kernelLoadBullet __device_func__ ( dev_den, m_vectGPUDens[ind], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_u, m_vectGPUVelU[ind], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_v, m_vectGPUVelV[ind], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_w, m_vectGPUVelW[ind], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_obs, m_vectGPUObst[ind], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function SetCurrentNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

#define _loadLeftFace(bul,grid)  kernelLoadLeftFace __device_func__ ( bul, grid, BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _loadRightFace(bul,grid) kernelLoadRightFace __device_func__ ( bul, grid, BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _loadUpFace(bul,grid) kernelLoadUpFace __device_func__ ( bul, grid, BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _loadDownFace(bul,grid) kernelLoadDownFace __device_func__ ( bul, grid, BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _loadFrontFace(bul,grid) kernelLoadFrontFace __device_func__ ( bul, grid, BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _loadBackFace(bul,grid) kernelLoadBackFace __device_func__ ( bul, grid, BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z )

	if ( m_link[ind]->ptrLeft not_eq nullptr )
	{
		_loadLeftFace( dev_den, m_vectGPUDens[ix(i-1,j,k,NODES_X,NODES_Y)] );
		_loadLeftFace( dev_u, m_vectGPUVelU[ix(i-1,j,k,NODES_X,NODES_Y)] );
		_loadLeftFace( dev_v, m_vectGPUVelV[ix(i-1,j,k,NODES_X,NODES_Y)] );
		_loadLeftFace( dev_w, m_vectGPUVelW[ix(i-1,j,k,NODES_X,NODES_Y)] );
		_loadLeftFace( dev_obs, m_vectGPUObst[ix(i-1,j,k,NODES_X,NODES_Y)] );
	}
	if ( m_link[ind]->ptrRight not_eq nullptr )
	{
		_loadRightFace( dev_den, m_vectGPUDens[ix(i+1,j,k,NODES_X,NODES_Y)] );
		_loadRightFace( dev_u, m_vectGPUVelU[ix(i+1,j,k,NODES_X,NODES_Y)] );
		_loadRightFace( dev_v, m_vectGPUVelV[ix(i+1,j,k,NODES_X,NODES_Y)] );
		_loadRightFace( dev_w, m_vectGPUVelW[ix(i+1,j,k,NODES_X,NODES_Y)] );
		_loadRightFace( dev_obs, m_vectGPUObst[ix(i+1,j,k,NODES_X,NODES_Y)] );
	}
	if ( m_link[ind]->ptrUp not_eq nullptr )
	{
		_loadUpFace( dev_den, m_vectGPUDens[ix(i,j+1,k,NODES_X,NODES_Y)] );
		_loadUpFace( dev_u, m_vectGPUVelU[ix(i,j+1,k,NODES_X,NODES_Y)] );
		_loadUpFace( dev_v, m_vectGPUVelV[ix(i,j+1,k,NODES_X,NODES_Y)] );
		_loadUpFace( dev_w, m_vectGPUVelW[ix(i,j+1,k,NODES_X,NODES_Y)] );
		_loadUpFace( dev_obs, m_vectGPUObst[ix(i,j+1,k,NODES_X,NODES_Y)] );
	}
	if ( m_link[ind]->ptrDown not_eq nullptr )
	{
		_loadDownFace( dev_den, m_vectGPUDens[ix(i,j-1,k,NODES_X,NODES_Y)] );
		_loadDownFace( dev_u, m_vectGPUVelU[ix(i,j-1,k,NODES_X,NODES_Y)] );
		_loadDownFace( dev_v, m_vectGPUVelV[ix(i,j-1,k,NODES_X,NODES_Y)] );
		_loadDownFace( dev_w, m_vectGPUVelW[ix(i,j-1,k,NODES_X,NODES_Y)] );
		_loadDownFace( dev_obs, m_vectGPUObst[ix(i,j-1,k,NODES_X,NODES_Y)] );
	}
	if ( m_link[ind]->ptrFront not_eq nullptr )
	{
		_loadFrontFace( dev_den, m_vectGPUDens[ix(i,j,k+1,NODES_X,NODES_Y)] );
		_loadFrontFace( dev_u, m_vectGPUVelU[ix(i,j,k+1,NODES_X,NODES_Y)] );
		_loadFrontFace( dev_v, m_vectGPUVelV[ix(i,j,k+1,NODES_X,NODES_Y)] );
		_loadFrontFace( dev_w, m_vectGPUVelW[ix(i,j,k+1,NODES_X,NODES_Y)] );
		_loadFrontFace( dev_obs, m_vectGPUObst[ix(i,j,k+1,NODES_X,NODES_Y)] );
	}
	if ( m_link[ind]->ptrBack not_eq nullptr )
	{
		_loadBackFace( dev_den, m_vectGPUDens[ix(i,j,k-1,NODES_X,NODES_Y)] );
		_loadBackFace( dev_u, m_vectGPUVelU[ix(i,j,k-1,NODES_X,NODES_Y)] );
		_loadBackFace( dev_v, m_vectGPUVelV[ix(i,j,k-1,NODES_X,NODES_Y)] );
		_loadBackFace( dev_w, m_vectGPUVelW[ix(i,j,k-1,NODES_X,NODES_Y)] );
		_loadBackFace( dev_obs, m_vectGPUObst[ix(i,j,k-1,NODES_X,NODES_Y)] );
	}

#undef _loadLeftFace(bul,grid)
#undef _loadRightFace(bul,grid)
#undef _loadUpFace(bul,grid)
#undef _loadDownFace(bul,grid)
#undef _loadFrontFace(bul,grid)
#undef _loadBackFace(bul,grid)

	if ( m_scHelper.GetCUDALastError( "call member function SetCurrentNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::GetCurrentNode( int i, int j, int k )
{
	GridsParamDim();

	int ind = ix( i, j, k, NODES_X, NODES_Y );

	/* exit node from bullet */
	kernelExitBullet __device_func__ ( m_vectNewDens[ind], dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelU[ind], dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelV[ind], dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelW[ind], dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	kernelPickData __device_func__ ( m_ptrDeviceVisual, m_vectNewDens[ind], i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function GetCurrentNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};
#endif