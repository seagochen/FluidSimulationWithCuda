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

inline __device__ __host__ int ix( cint x, cint tilex)
{
	if ( x >= tilex or x < 0 ) return -1;
	return x;
};

inline __device__ __host__ int ix( cint i, cint j, cint tilex, cint tiley )
{
	if ( i < 0 or j < 0 ) return -1;

	int x; if ( i >= tilex ) x = tilex - 1;
	int y; if ( j >= tiley ) y = tiley - 1;

	x = i; y = j;
	return x + y * tilex;
};

inline __host__ __device__ int ix
	( cint i, cint j, cint k, cint tilex, cint tiley, cint tilez )
{
	if ( i < 0 or i >= tilex ) return -1;
	if ( j < 0 or j >= tiley ) return -1;
	if ( k < 0 or k >= tilez ) return -1;

	return i + j * tilex + k * tilex * tiley;
};

extern // updated: 2014/3/28
__global__ void kernelSetBound( double *dst, cint tilex, cint tiley, cint tilez );

extern // updated: 2014/3/28
__global__ void kernelDeassemble( double *dst, cdouble *src, 
								 cint srcx, cint srcy, cint srcz, 
								 cint dstx, cint dsty, cint dstz, 
								 cint offi, cint offj, cint offk );

extern // updated: 2014/3/28
__global__ void kernelGlobalToBullet( double *dst, cdouble *src,
									 cint srcx, cint srcy, cint srcz, 
									 cint dstx, cint dsty, cint dstz, 
									 cint grdx, cint grdy, cint grdz,
									 cint offi, cint offj, cint offk );


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

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectCompBufs, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), COMP_BUFS ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectBulletBufs, 
		BULLET_X * BULLET_Y * BULLET_Z * sizeof(double), BUL_BUFS ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectBigBufs,
		VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(double), BIG_BUFS ) ) goto Error;



	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectSmallDens, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectSmallVelU, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectSmallVelV, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;

	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectSmallVelW, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), NODES_X * NODES_Y * NODES_Z ) ) goto Error;



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
		cout << "size of m_vectBulletBufs: " << m_vectBulletBufs.size() << endl
			<< "size of m_vectCompBufs: " << m_vectCompBufs.size() << endl;
		
		cout << "size of m_vectSmallDens: " << m_vectSmallDens.size() << endl
			<< "size of m_vectSmallVelU: " << m_vectSmallVelU.size() << endl
			<< "size of m_vectSmallVelV: " << m_vectSmallVelV.size() << endl
			<< "size of m_vectSmallVelW: " << m_vectSmallVelW.size() << endl;
};

void FluidSimProc::FreeResource( void )
{
	for ( int i = 0; i < m_vectBulletBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectBulletBufs[i] );

	for ( int i = 0; i < m_vectCompBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectCompBufs[i] );

	for ( int i = 0; i < m_vectBigBufs.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectBigBufs[i] );

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		m_scHelper.FreeDeviceBuffers( 1, &m_vectSmallDens[i] );
		m_scHelper.FreeDeviceBuffers( 1, &m_vectSmallVelU[i] );
		m_scHelper.FreeDeviceBuffers( 1, &m_vectSmallVelV[i] );
		m_scHelper.FreeDeviceBuffers( 1, &m_vectSmallVelW[i] );
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

void FluidSimProc::ClearBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 33, 22, BULLET_X, BULLET_Y, BULLET_Z );
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
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelSetBound __device_func__ ( comp_obst, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelLoadBullet __device_func__
		( dev_obs, comp_obst,  BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function InitBound failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( not fluid->run ) return;
	
	SolveGlobal( DELTATIME, true, true, true );

	SolveLocal( DELTATIME/2.f, true, true );

	GenerVolumeImg();

	RefreshStatus( fluid );
};

void FluidSimProc::GenerVolumeImg( void )
{
#if 0

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
	for ( int i = 0; i < NODES_X; i++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int k = 0; k < NODES_Z; k++ )
	{
		kernelPickData __device_func__ ( m_ptrDeviceVisual, m_vectSmallDens[ix(i,j,k,NODES_X,NODES_Y)], 
			VOLUME_X, VOLUME_Y, VOLUME_Z,
			GRIDS_X, GRIDS_Y, GRIDS_Z, 
			i, j, k );
	}

#else

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, VOLUME_X, VOLUME_Y, VOLUME_Z );
	kernelPickData __device_func__ ( m_ptrDeviceVisual, big_den, VOLUME_X, VOLUME_Y, VOLUME_Z );
	
	for ( int i = 0; i < NODES_X; i++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int k = 0; k < NODES_Z; k++ )
	{
		if ( j > 0 )
		{
			m_scHelper.DeviceParamDim
				( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
			kernelPickData __device_func__
				( m_ptrDeviceVisual, m_vectSmallDens[ix(i, j, k, NODES_X, NODES_Y)], 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z, 
				i, j, k );
		}
	}

#endif
};

void FluidSimProc::SolveGlobal( cdouble dt, bool add, bool vel, bool dens )
{
	if ( add ) SourceSolverGlobal( dt );
	if ( vel ) VelocitySolverGlobal( dt );
	if ( dens ) DensitySolverGlobal( dt );

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_X, GRIDS_X );

	kernelExitBullet __device_func__
		( comp_den, dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	kernelExitBullet __device_func__
		( comp_u, dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	kernelExitBullet __device_func__
		( comp_v, dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	kernelExitBullet __device_func__
		( comp_w, dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
};

void FluidSimProc::SolveLocal( cdouble dt, bool vel, bool dens )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, VOLUME_X, VOLUME_Y, VOLUME_Z );

	kernelUpScalingInterpolation __device_func__ ( big_den, comp_den, 
		GRIDS_X, GRIDS_Y, GRIDS_Z, 
		VOLUME_X, VOLUME_Y, VOLUME_Z,
		2, 2, 2 );

	kernelUpScalingInterpolation __device_func__ ( big_u, comp_u, 
		GRIDS_X, GRIDS_Y, GRIDS_Z, 
		VOLUME_X, VOLUME_Y, VOLUME_Z,
		2, 2, 2 );

	kernelUpScalingInterpolation __device_func__ ( big_v, comp_v, 
		GRIDS_X, GRIDS_Y, GRIDS_Z, 
		VOLUME_X, VOLUME_Y, VOLUME_Z,
		2, 2, 2 );

	kernelUpScalingInterpolation __device_func__ ( big_w, comp_w, 
		GRIDS_X, GRIDS_Y, GRIDS_Z, 
		VOLUME_X, VOLUME_Y, VOLUME_Z,
		2, 2, 2 );

#if 0

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k = 0; k < NODES_Z; k++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int i = 0; i < NODES_X; i++ )
	{
		kernelDeassemble __device_func__ ( m_vectSmallDens[ix(i,j,k,NODES_X,NODES_Y)], big_den, 
			VOLUME_X, VOLUME_Y, VOLUME_Z,
			GRIDS_X, GRIDS_Y, GRIDS_Z,
			i, j, k );
	}

//#else

	for ( int i = 0; i < NODES_X; i++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int k = 0; k < NODES_Z; k++ )
	{
		if ( j > 0 )
		{
			m_scHelper.DeviceParamDim
				( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
			
			kernelDeassemble __device_func__ ( m_vectSmallDens[ix(i, j, k, NODES_X, NODES_Y)], big_den, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );
			
			kernelDeassemble __device_func__ ( m_vectSmallVelU[ix(i, j, k, NODES_X, NODES_Y)], big_u, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );

			kernelDeassemble __device_func__ ( m_vectSmallVelV[ix(i, j, k, NODES_X, NODES_Y)], big_v, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );

			kernelDeassemble __device_func__ ( m_vectSmallVelW[ix(i, j, k, NODES_X, NODES_Y)], big_w, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );
		}
	}
#endif

	for ( int i = 0; i < NODES_X; i++ ) for ( int j = 0; j < NODES_Y; j++ ) for ( int k = 0; k < NODES_Z; k++ )
	{
		if ( j > 0 )
		{
			m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 33, 22, BULLET_X, BULLET_Y, BULLET_Z );

			kernelGlobalToBullet __device_func__ ( loc_den, dev_den, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				BULLET_X, BULLET_Y, BULLET_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );

			kernelGlobalToBullet __device_func__ ( loc_u, dev_u, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				BULLET_X, BULLET_Y, BULLET_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );

			kernelGlobalToBullet __device_func__ ( loc_v, dev_v, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				BULLET_X, BULLET_Y, BULLET_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );

			kernelGlobalToBullet __device_func__ ( loc_w, dev_w, 
				VOLUME_X, VOLUME_Y, VOLUME_Z,
				BULLET_X, BULLET_Y, BULLET_Z,
				GRIDS_X, GRIDS_Y, GRIDS_Z,
				i, j, k );

			m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

			if ( vel ) VelocitySolverLocal( dt );
			if ( dens ) DensitySolverLocal( dt );

			kernelExitBullet __device_func__ 
				( m_vectSmallDens[ix(i, j, k, NODES_X, NODES_Y)], loc_den,
				GRIDS_X, GRIDS_Y, GRIDS_Z, 
				BULLET_X, BULLET_Y, BULLET_Z );
		}
	}
};