/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 24, 2014
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

void FluidSimProc::CreateTopology( void )
{
	for ( int k = 0; k < GNODES_X; k++ )
	{
		for ( int j = 0; j < GNODES_X; j++ )
		{
			for ( int i = 0; i < GNODES_X; i++ )
			{
				/* left */
				if ( i >= 1 )
					gpu_node[cudaIndex3D( i, j, k, GNODES_X )]->ptrLeft  = gpu_node[cudaIndex3D( i-1, j, k, GNODES_X )];
				/* right */
				if ( i <= GNODES_X - 2 )
					gpu_node[cudaIndex3D( i, j, k, GNODES_X )]->ptrRight = gpu_node[cudaIndex3D( i+1, j, k, GNODES_X )];
				/* down */
				if ( j >= 1 )
					gpu_node[cudaIndex3D( i, j, k, GNODES_X )]->ptrDown  = gpu_node[cudaIndex3D( i, j-1, k, GNODES_X )];
				/* up */
				if ( j <= GNODES_X - 2 )
					gpu_node[cudaIndex3D( i, j, k, GNODES_X )]->ptrUp    = gpu_node[cudaIndex3D( i, j+1, k, GNODES_X )];
				/* back */
				if ( k >= 1 )
					gpu_node[cudaIndex3D( i, j, k, GNODES_X )]->ptrBack  = gpu_node[cudaIndex3D( i, j, k-1, GNODES_X )];
				/* front */
				if ( k <= GNODES_X - 2 )
					gpu_node[cudaIndex3D( i, j, k, GNODES_X )]->ptrFront = gpu_node[cudaIndex3D( i, j, k+1, GNODES_X )];
			}
		}
	}

	for ( int k = 0; k < HNODES_X; k++ )
	{
		for ( int j = 0; j < HNODES_X; j++ )
		{
			for ( int i = 0; i < HNODES_X; i++ )
			{
				/* left */
				if ( i >= 1 )
					host_node[cudaIndex3D( i, j, k, HNODES_X )]->ptrLeft  = host_node[cudaIndex3D( i-1, j, k, HNODES_X )];
				/* right */
				if ( i <= HNODES_X - 2 )
					host_node[cudaIndex3D( i, j, k, HNODES_X )]->ptrRight = host_node[cudaIndex3D( i+1, j, k, HNODES_X )];
				/* down */
				if ( j >= 1 )
					host_node[cudaIndex3D( i, j, k, HNODES_X )]->ptrDown  = host_node[cudaIndex3D( i, j-1, k, HNODES_X )];
				/* up */
				if ( j <= HNODES_X - 2 )
					host_node[cudaIndex3D( i, j, k, HNODES_X )]->ptrUp    = host_node[cudaIndex3D( i, j+1, k, HNODES_X )];
				/* back */
				if ( k >= 1 )
					host_node[cudaIndex3D( i, j, k, HNODES_X )]->ptrBack  = host_node[cudaIndex3D( i, j, k-1, HNODES_X )];
				/* front */
				if ( k <= HNODES_X - 2 )
					host_node[cudaIndex3D( i, j, k, HNODES_X )]->ptrFront = host_node[cudaIndex3D( i, j, k+1, HNODES_X )];

				host_node[cudaIndex3D( i, j, k, HNODES_X )]->x = i;
				host_node[cudaIndex3D( i, j, k, HNODES_X )]->y = j;
				host_node[cudaIndex3D( i, j, k, HNODES_X )]->z = k;
			}
		}
	}
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

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ ) host_node[i]->updated = false;

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
	if ( cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		helper.GetCUDALastError( "host function: cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::InitBoundary( void )
{
	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	/* zero boundary buffers */
	kernelZeroBuffers __device_func__ ( dev_obs, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int i = 0; i < host_obstacle.size(); i++ )
	{
		if ( cudaMemcpy( host_obstacle[i], dev_obs, m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}
	}

	kernelCopyGrids __device_func__ ( gd_obstacle, dev_obs );

	// TODO more boundary condition
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;
	
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	kernelCopyGrids __device_func__ ( dev_den, gd_density );
	kernelCopyGrids __device_func__ ( dev_u, gd_velocity_u );
	kernelCopyGrids __device_func__ ( dev_v, gd_velocity_v );
	kernelCopyGrids __device_func__ ( dev_w, gd_velocity_w );

	SolveNavierStokesEquation( DELTATIME, true );

	double rate = 1.f/(double)HNODES_X;
		
	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		ptr = host_node[cudaIndex3D(i,j,k,HNODES_X)];

		kernelInterRootGrids __device_func__ ( dev_density[cudaIndex3D(i,j,k,HNODES_X)], dev_den, i, j, k, rate );
		kernelInterRootGrids __device_func__ ( dev_velocity_u[cudaIndex3D(i,j,k,HNODES_X)], dev_u, i, j, k, rate );
		kernelInterRootGrids __device_func__ ( dev_velocity_v[cudaIndex3D(i,j,k,HNODES_X)], dev_v, i, j, k, rate );
		kernelInterRootGrids __device_func__ ( dev_velocity_w[cudaIndex3D(i,j,k,HNODES_X)], dev_w, i, j, k, rate );
	}

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		ptr = host_node[i];
		kernelInterLeafGrids __device_func__ ( gd_density, dev_density[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_u, dev_velocity_u[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_v, dev_velocity_v[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_w, dev_velocity_w[i], ptr->x, ptr->y, ptr->z, rate );
	}

	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		kernelPickData __device_func__ ( dev_visual, dev_density[cudaIndex3D(i,j,k,HNODES_X)], i, j, k, GRIDS_X );
	}

	RefreshStatus( fluid );
};

void FluidSimProc::ClearBuffers( void )
{
	DeviceParamDim();

	_zero( gd_density );
	_zero( gd_velocity_u );
	_zero( gd_velocity_v );
	_zero( gd_velocity_w );

	for ( int i = 0; i < m_vectCompBufs.size(); i++ ) _zero( m_vectCompBufs[i] );
		 	 

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		_zero( m_vectGPUDens[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelU[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelV[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelW[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUObst[i] ); _zero( m_vectNewDens[i] );
	}

	if ( helper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};