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
	/* set nodes for gpu and device */
	size_t size = GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double);
	size_t node = NODES_X * NODES_Y * NODES_Z;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostDens, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelU, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelV, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelW, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostObst, size, node ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevDens, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevVelU, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevVelV, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevVelW, size, node ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevObst, size, node ) ) goto Error;

	/* set computational buffers */
	size = COMPS_X * COMPS_Y * COMPS_Z * sizeof(double);
	node = COMP_BUFS;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectCompBufs, size, node ) ) goto Error;

	/* set bullet buffers */
	size = BULLET_X * BULLET_Y * BULLET_Z * sizeof(double);
	node = BULLET_BUFS;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectBulletBufs, size, node ) ) goto Error;

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
		m_scHelper.FreeDeviceBuffers( 5, 
			&m_vectDevDens[i], &m_vectDevVelU[i], &m_vectDevVelV[i], &m_vectDevVelW[i], &m_vectDevObst[i] );

		m_scHelper.FreeHostBuffers( 5,
			&m_vectHostDens[i], &m_vectHostVelU[i], &m_vectHostVelV[i], &m_vectHostVelW[i], &m_vectHostObst[i] );
	}


	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
		SAFE_DELT_PTR( m_link[i] );

	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectCompBufs[i] );

	for ( int i = 0; i < m_vectBulletBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectBulletBufs[i] );

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

void FluidSimProc::ClearBuffers( void )
{
	GridsParamDim();
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectDevDens[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectDevVelU[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectDevVelV[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectDevVelW[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	CompParamDim();
	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectCompBufs[i], COMPS_X, COMPS_Y, COMPS_Z );

	BulletParamDim();
	for ( int i = 0; i < m_vectBulletBufs.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectBulletBufs[i], BULLET_X, BULLET_Y, BULLET_Z );

	VisualParamDim();
	kernelZeroBuffers __device_func__ ( m_ptrDeviceVisual, VOLUME_X, VOLUME_Y, VOLUME_Z );

	if ( m_scHelper.GetCUDALastError( "call member function ZeroBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::LoadPreStage( void )
{
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		cudaMemcpy( m_vectDevDens[i], m_vectHostDens[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectDevVelU[i], m_vectHostVelU[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectDevVelV[i], m_vectHostVelV[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectDevVelW[i], m_vectHostVelW[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
		cudaMemcpy( m_vectDevObst[i], m_vectHostObst[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyHostToDevice );
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
		cudaMemcpy( m_vectHostDens[i], m_vectDevDens[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelU[i], m_vectDevVelU[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelV[i], m_vectDevVelV[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelW[i], m_vectDevVelW[i], GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), cudaMemcpyDeviceToHost );
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

	for ( int n = 0; n < NODES_X * NODES_Y * NODES_Z; n++ )
	{
		cudaMemcpy( m_vectDevObst[n], m_vectHostObst[n], 
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

	GridsParamDim();
	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelAssembleCompBufs __device_func__ ( comp_dens, COMPS_X, COMPS_Y, COMPS_Z,
			m_vectDevDens[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ ( comp_velu, COMPS_X, COMPS_Y, COMPS_Z,
			m_vectDevVelU[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ ( comp_velv, COMPS_X, COMPS_Y, COMPS_Z,
			m_vectDevVelV[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ ( comp_velw, COMPS_X, COMPS_Y, COMPS_Z,
			m_vectDevVelW[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ ( comp_obst, COMPS_X, COMPS_Y, COMPS_Z,
			m_vectDevObst[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );
	}


	CompParamDim();
	kernelLoadBullet __device_func__ ( dev_u, comp_velu, BULLET_X, BULLET_Y, BULLET_Z, COMPS_X, COMPS_Y, COMPS_Z );
	kernelLoadBullet __device_func__ ( dev_v, comp_velv, BULLET_X, BULLET_Y, BULLET_Z, COMPS_X, COMPS_Y, COMPS_Z );
	kernelLoadBullet __device_func__ ( dev_w, comp_velw, BULLET_X, BULLET_Y, BULLET_Z, COMPS_X, COMPS_Y, COMPS_Z );
	kernelLoadBullet __device_func__ ( dev_den, comp_dens, BULLET_X, BULLET_Y, BULLET_Z, COMPS_X, COMPS_Y, COMPS_Z );
	kernelLoadBullet __device_func__ ( dev_obs, comp_obst, BULLET_X, BULLET_Y, BULLET_Z, COMPS_X, COMPS_Y, COMPS_Z );


	CompParamDim();
	SolveNavierStokesEquation( DELTATIME, true, true, true );


	CompParamDim();
	kernelExitBullet __device_func__ ( comp_velu, dev_u, COMPS_X, COMPS_Y, COMPS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( comp_velv, dev_v, COMPS_X, COMPS_Y, COMPS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( comp_velw, dev_w, COMPS_X, COMPS_Y, COMPS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( comp_dens, dev_den, COMPS_X, COMPS_Y, COMPS_Z, BULLET_X, BULLET_Y, BULLET_Z );


	GridsParamDim();
	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelDeassembleCompBufs __device_func__ ( m_vectDevDens[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			comp_dens, COMPS_X, COMPS_Y, COMPS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelDeassembleCompBufs __device_func__ ( m_vectDevVelU[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			comp_velu, COMPS_X, COMPS_Y, COMPS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelDeassembleCompBufs __device_func__ ( m_vectDevVelV[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			comp_velv, COMPS_X, COMPS_Y, COMPS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );

		kernelDeassembleCompBufs __device_func__ ( m_vectDevVelW[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			comp_velw, COMPS_X, COMPS_Y, COMPS_Z,
			i * GRIDS_X, j * GRIDS_Y, k * GRIDS_Z,
			1.f, 1.f, 1.f );
	}

#if 1
	GridsParamDim();
	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
//		kernelPickData __device_func__
//			( m_ptrDeviceVisual, m_vectDevDens[ix(i,j,k,NODES_X,NODES_Y)], i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z );

		kernelPickData __device_func__ ( m_ptrDeviceVisual, VOLUME_X, VOLUME_Y, VOLUME_Z,
			m_vectDevDens[ix(i,j,k,NODES_X,NODES_Y)], GRIDS_X, GRIDS_Y, GRIDS_Z,
			i, j, k, 1.f, 1.f, 1.f );
	}

#else
	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelInterLeafGrids __device_func__
			( global_dens, m_vectDevDens[ix(i,j,k,NODES_X,NODES_Y)], i, j, k, 0.5f );
	}
	
	kernelPickData __device_func__
		( m_ptrDeviceVisual, global_dens, 0, 0, 0, GRIDS_X, GRIDS_Y, GRIDS_Z );
#endif

	RefreshStatus( fluid );
};

void FluidSimProc::SolveNavierStokesEquation( cdouble dt, bool add, bool vel, bool dens )
{
	if ( add ) SourceSolver( dt );
	if ( vel ) VelocitySolver( dt );
	if ( dens ) DensitySolver( dt );
};