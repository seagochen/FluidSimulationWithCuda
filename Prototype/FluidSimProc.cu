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
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostDens, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelU, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelV, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostVelW, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForHost( &m_vectHostObst, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;


	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectToyDens, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectToyVelU, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectToyVelV, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectToyVelW, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectToyObst, 
		TOY_X * TOY_Y * TOY_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;


	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectCompBufs, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 5 ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectBulletBufs, 
		BULLET_X * BULLET_Y * BULLET_Z * sizeof(double), COMP_BUFS ) ) goto Error;

	m_scHelper.CreateDeviceBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrDeviceVisual );
	m_scHelper.CreateHostBuffers( VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR), 1, &m_ptrHostVisual );

	m_scHelper.CreateHostBuffers( TOY_X * TOY_Y * TOY_Z * sizeof(double), 1, &m_ptrHostToy );
	m_scHelper.CreateHostBuffers( GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 1, &m_ptrHostComp );
	m_scHelper.CreateHostBuffers( BULLET_X * BULLET_Y * BULLET_Z * sizeof(double), 1, &m_ptrHostBullet );

	goto Success;

Error:
		cout << "create computation buffers device failed" << endl;
		FreeResource();
		exit(1);

Success:
		cout << "size of m_vectBulletBufs: " << m_vectBulletBufs.size() << endl
			<< "size of m_vectCompBufs: " << m_vectCompBufs.size() << endl
			<< "size of m_vectToyDens: "  << m_vectToyDens.size() << endl
			<< "size of m_vectToyVelU: "  << m_vectToyVelU.size() << endl
			<< "size of m_vectToyVelV: "  << m_vectToyVelV.size() << endl
			<< "size of m_vectToyVelW: "  << m_vectToyVelW.size() << endl
			<< "size of m_vectToyObst: "  << m_vectToyObst.size() << endl;

		cout << "size of m_vectHostDens: " << m_vectHostDens.size() << endl
			<< "size of m_vectHostVelU: " << m_vectHostVelU.size() << endl
			<< "size of m_vectHostVelV: " << m_vectHostVelV.size() << endl
			<< "size of m_vectHostVelW: " << m_vectHostVelW.size() << endl
			<< "size of m_vectHostObst: " << m_vectHostObst.size() << endl;
};

void FluidSimProc::FreeResource( void )
{
	for ( int i = 0; i < m_vectBulletBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectBulletBufs[i] );

	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectCompBufs[i] );

	m_scHelper.FreeHostBuffers( 5, &m_vectHostDens[0], &m_vectHostVelU[0],
		&m_vectHostVelV[0], &m_vectHostVelW[0], &m_vectHostObst[0] );

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		m_scHelper.FreeDeviceBuffers( 5, &m_vectToyDens[i], &m_vectToyVelU[i],
			&m_vectToyVelV[i], &m_vectToyVelW[i], &m_vectToyObst[i] );
	}

	m_scHelper.FreeDeviceBuffers( 1, &m_ptrDeviceVisual );
	m_scHelper.FreeHostBuffers( 1, &m_ptrHostVisual );

	m_scHelper.FreeHostBuffers( 3, &m_ptrHostToy, &m_ptrHostComp, &m_ptrHostBullet );
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
	
	AssembleBuffers();

	SolveNavierStokesEquation( DELTATIME, true );

	DeassembleBuffers();

	GenerVolumeImg();

	RefreshStatus( fluid );
};

void FluidSimProc::GenerVolumeImg( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, TOY_X, TOY_Y, TOY_Z );
	for ( int i = 0; i < NODES_X; i++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int k = 0; k < NODES_Z; k++ )
	{
		kernelPickData __device_func__ ( m_ptrDeviceVisual, VOLUME_X, VOLUME_Y, VOLUME_Z,
			m_vectToyDens[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z,
			i, j, k, 1.f, 1.f, 1.f );
	}
};

void FluidSimProc::ClearBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, TOY_X, TOY_Y, TOY_Z );
	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectToyDens[i], TOY_X, TOY_Y, TOY_Z );
		kernelZeroBuffers __device_func__ ( m_vectToyVelU[i], TOY_X, TOY_Y, TOY_Z );
		kernelZeroBuffers __device_func__ ( m_vectToyVelV[i], TOY_X, TOY_Y, TOY_Z );
		kernelZeroBuffers __device_func__ ( m_vectToyVelW[i], TOY_X, TOY_Y, TOY_Z );
	}


	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 22, 33, BULLET_X, BULLET_Y, BULLET_Z );
	for ( int i = 0; i < m_vectBulletBufs.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectBulletBufs[i], BULLET_X, BULLET_Y, BULLET_Z );


	if ( m_scHelper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::InitBoundary( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, TOY_X, TOY_Y, TOY_Z );
	for ( int i = 0; i < m_vectToyObst.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectToyObst[i], TOY_X, TOY_Y, TOY_Z );

	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	for ( int i = 0; i < m_vectHostObst.size(); i++ )
		cudaMemcpy( m_vectHostObst[i], m_vectToyObst[i], 
		sizeof(double) * TOY_X * TOY_Y * TOY_Z, cudaMemcpyDeviceToHost );

	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	cint halfx = TOY_X / 2;
	cint halfz = TOY_Z / 2;

	for ( int k = 0; k < TOY_Z; k++ )
	{
		for ( int j = 0; j < TOY_Y; j++ )
		{
			for ( int i = 0; i < TOY_X; i++ )
			{					
				if ( j < 3 and 
					i >= halfx - 2 and i < halfx + 2 and 
					k >= halfz - 2 and k < halfz + 2 )
					m_vectHostObst[0][ix( i, j, k, TOY_X, TOY_Y )] = MACRO_BOUNDARY_SOURCE;
			}
		}
	}


	for ( int n = 0; n < NODES_X * NODES_Y * NODES_Z; n++ )
	{
		cudaMemcpy( m_vectToyObst[n], m_vectHostObst[n], 
			sizeof(double) * TOY_X * TOY_Y * TOY_Z, cudaMemcpyHostToDevice );
	}

	if ( m_scHelper.GetCUDALastError( "call member function InitBound failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::AssembleBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, TOY_X, TOY_Y, TOY_Z );

	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelAssembleCompBufs __device_func__ (
			dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z, 
			m_vectToyDens[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z, 
			i, j, k,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ (
			dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z, 
			m_vectToyVelU[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z, 
			i, j, k,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ (
			dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, 
			m_vectToyVelV[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z, 
			i, j, k,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ (
			dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z, 
			m_vectToyVelW[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z, 
			i, j, k,
			1.f, 1.f, 1.f );

		kernelAssembleCompBufs __device_func__ (
			dev_obs, GRIDS_X, GRIDS_Y, GRIDS_Z, 
			m_vectToyObst[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z, 
			i, j, k,
			1.f, 1.f, 1.f );
	}

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function InitBound failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::DeassembleBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, TOY_X, TOY_Y, TOY_Z );

	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelDeassembleCompBufs __device_func__ (
			m_vectToyDens[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z,
			dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z,
			i, j, k, 
			1.f, 1.f, 1.f );

		kernelDeassembleCompBufs __device_func__ (
			m_vectToyVelU[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z,
			dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z,
			i, j, k, 
			1.f, 1.f, 1.f );

		kernelDeassembleCompBufs __device_func__ (
			m_vectToyVelV[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z,
			dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z,
			i, j, k, 
			1.f, 1.f, 1.f );

		kernelDeassembleCompBufs __device_func__ (
			m_vectToyVelW[ix(i,j,k,NODES_X,NODES_Y)], TOY_X, TOY_Y, TOY_Z,
			dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z,
			i, j, k, 
			1.f, 1.f, 1.f );
	}
};