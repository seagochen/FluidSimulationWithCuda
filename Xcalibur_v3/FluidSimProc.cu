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

using namespace sge;

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




/*******************************************************************************************/


#if 0
void FluidSimProc::HostToDevice( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{
		cudaMemcpy( dev_density[i], host_density[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velocity_u[i], host_velocity_u[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velocity_v[i], host_velocity_v[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velocity_w[i], host_velocity_w[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_obstacle[i], host_obstacle[i], m_node_size, cudaMemcpyHostToDevice );
	}
};

void FluidSimProc::DeviceToHost( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{
		cudaMemcpy( host_density[i], dev_density[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_u[i], dev_velocity_u[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_velocity_v[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_velocity_w[i], m_node_size, cudaMemcpyDeviceToHost );
	}
};

void FluidSimProc::clearBullet( void )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int i = 11; i < dev_buffers_num; i++ )
	{
		kernelZeroBuffers __device_func__ ( dev_buffers[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( m_scHelper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickNodeToBullet( int i, int j, int k )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	/* upload center node to GPU device */
	kernelCopyBuffers __device_func__ ( dev_u, node_velocity_u[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_v, node_velocity_v[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_w, node_velocity_w[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_den,  node_density[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_obs, node_obstacle[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickNeighborsToBullet( int i, int j, int k )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	/* upload neighbouring buffers to GPU device */
	ptr = gpu_node[Index( i, j, k, GNODES_X )];
	if ( ptr->ptrLeft not_eq nullptr )
	{
		kernelCopyBuffers __device_func__ ( velu_L, node_velocity_u[Index( i-1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velv_L, node_velocity_v[Index( i-1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velw_L, node_velocity_w[Index( i-1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dens_L,    node_density[Index( i-1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( ptr->ptrRight not_eq nullptr )
	{
		kernelCopyBuffers __device_func__ ( velu_R, node_velocity_u[Index( i+1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velv_R, node_velocity_v[Index( i+1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velw_R, node_velocity_w[Index( i+1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dens_R,    node_density[Index( i+1, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( ptr->ptrUp not_eq nullptr )
	{
		kernelCopyBuffers __device_func__ ( velu_U, node_velocity_u[Index( i, j+1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velv_U, node_velocity_v[Index( i, j+1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velw_U, node_velocity_w[Index( i, j+1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dens_U,    node_density[Index( i, j+1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( ptr->ptrDown not_eq nullptr )
	{
		kernelCopyBuffers __device_func__ ( velu_D, node_velocity_u[Index( i, j-1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velv_D, node_velocity_v[Index( i, j-1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velw_D, node_velocity_w[Index( i, j-1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dens_D,    node_density[Index( i, j-1, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );

	}

	if ( ptr->ptrFront not_eq nullptr )
	{
		kernelCopyBuffers __device_func__ ( velu_F, node_velocity_u[Index( i, j, k+1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velv_F, node_velocity_v[Index( i, j, k+1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velw_F, node_velocity_w[Index( i, j, k+1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dens_F,    node_density[Index( i, j, k+1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );

	}

	if ( ptr->ptrBack not_eq nullptr )
	{
		kernelCopyBuffers __device_func__ ( velu_B, node_velocity_u[Index( i, j, k-1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velv_B, node_velocity_v[Index( i, j, k-1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( velw_B, node_velocity_w[Index( i, j, k-1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dens_B,    node_density[Index( i, j, k-1, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( m_scHelper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::LoadBullet( int i, int j, int k )
{
	clearBullet();
	pickNodeToBullet( i, j, k );
	pickNeighborsToBullet( i, j, k );
};

void FluidSimProc::pickBulletToNode( int i, int j, int k )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelCopyBuffers __device_func__ ( node_velocity_u[Index(i,j,k,GNODES_X)], velu_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( node_velocity_v[Index(i,j,k,GNODES_X)], velv_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( node_velocity_w[Index(i,j,k,GNODES_X)], velw_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( node_density[Index(i,j,k,GNODES_X)],    dens_C, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickImgFromNode( int i, int j, int k )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	i += m_cursor.x;
	j += m_cursor.y;
	k += m_cursor.z;
	kernelPickData __device_func__( dev_visual, dev_den, i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "device kernel: kernelPickData failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::ExitBullet( int i, int j, int k )
{
	pickBulletToNode( i, j, k );
	pickImgFromNode( i, j, k );
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

	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ ) host_node[i]->updated = false;

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
		m_scHelper.GetCUDALastError( "host function: cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	zeroShareBuffers();
	zeroHostRes();
	SolveRootNode();
	//	SolveLeafNode();
	RefreshStatus( fluid );
};

void FluidSimProc::SolveRootNode( void )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	zeroTempoBuffers();

	kernelCopyBuffers __device_func__ ( dev_den, gd_density, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_u, gd_velocity_u, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_v, gd_velocity_v, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_w, gd_velocity_w, GRIDS_X, GRIDS_Y, GRIDS_Z );

	SolveNavierStokesEquation( DELTATIME, true );

	double rate = 1.f/(double)HNODES_X;

	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		ptr = host_node[Index(i,j,k,HNODES_X)];

		kernelInterRootGrids __device_func__ ( dev_density[Index(i,j,k,HNODES_X)], dev_den, i, j, k, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterRootGrids __device_func__ ( dev_velocity_u[Index(i,j,k,HNODES_X)], dev_u, i, j, k, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterRootGrids __device_func__ ( dev_velocity_v[Index(i,j,k,HNODES_X)], dev_v, i, j, k, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterRootGrids __device_func__ ( dev_velocity_w[Index(i,j,k,HNODES_X)], dev_w, i, j, k, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	/* error is here ! */
	//	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	//	{
	//		kernelCopyGrids __device_func__ ( dev_den, dev_density[Index(i,j,k,HNODES_X)] );
	//		kernelCopyGrids __device_func__ ( dev_u, dev_velocity_u[Index(i,j,k,HNODES_X)] );
	//		kernelCopyGrids __device_func__ ( dev_v, dev_velocity_v[Index(i,j,k,HNODES_X)] );
	//		kernelCopyGrids __device_func__ ( dev_w, dev_velocity_w[Index(i,j,k,HNODES_X)] );
	//
	//		kernelClearHalo __device_func__ ( dev_den );
	//		kernelClearHalo __device_func__ ( dev_u );
	//		kernelClearHalo __device_func__ ( dev_v );
	//		kernelClearHalo __device_func__ ( dev_w );
	//		
	//		SolveNavierStokesEquation( DELTATIME/2.f, false );
	//
	//		kernelCopyGrids __device_func__ ( dev_density[Index(i,j,k,HNODES_X)], dev_den );
	//		kernelCopyGrids __device_func__ ( dev_velocity_u[Index(i,j,k,HNODES_X)], dev_u );
	//		kernelCopyGrids __device_func__ ( dev_velocity_v[Index(i,j,k,HNODES_X)], dev_v );
	//		kernelCopyGrids __device_func__ ( dev_velocity_w[Index(i,j,k,HNODES_X)], dev_w );
	//	}

	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{
		ptr = host_node[i];
		kernelInterLeafGrids __device_func__ ( gd_density, dev_density[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterLeafGrids __device_func__ ( gd_velocity_u, dev_velocity_u[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterLeafGrids __device_func__ ( gd_velocity_v, dev_velocity_v[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterLeafGrids __device_func__ ( gd_velocity_w, dev_velocity_w[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		kernelPickData __device_func__ ( dev_visual, dev_density[Index(i,j,k,HNODES_X)], i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};

void FluidSimProc::SolveLeafNode( void )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	zeroTempoBuffers();

#if HNODES_X >= 3

	/* sum density */
	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		kernelSumDensity __device_func__ ( dev_dtpbuf, dev_density[Index(i,j,k,HNODES_X)], Index(i,j,k,HNODES_X) );
	}
	cudaMemcpy( host_dtpbuf, dev_dtpbuf, sizeof(double)*TPBUFFER_S, cudaMemcpyDeviceToHost );

	/* ignore the node with density lower than 1 */
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{
		if ( 1.f > host_dtpbuf[i] )
			host_node[i]->active = false;
		else
			host_node[i]->active = true;
	}

#endif

	/* move cursor */
	for ( int ck = 0; ck < CURSOR_X; ck ++ ) for ( int cj = 0; cj < CURSOR_X; cj++ ) for ( int ci = 0; ci < CURSOR_X; ci++ )
	{
		/* update cursor */
		m_cursor.x = ci;
		m_cursor.y = cj;
		m_cursor.z = ck;

		ReadBuffers();

		/* solve independent gpu nodes one by one */
		for ( int k = 0; k < GNODES_X; k++ ) for ( int j = 0; j < GNODES_X; j++ ) for ( int i = 0; i < GNODES_X; i++ )
		{
#if HNODES_X >= 3
			if ( !gpu_node[Index(i,j,k,GNODES_X)]->updated
				and gpu_node[Index(i,j,k,GNODES_X)]->active )
#else
			if ( !gpu_node[Index(i,j,k,GNODES_X)]->updated )
#endif
			{
				LoadBullet(i,j,k);

				kernelHandleHalo __device_func__ ( dev_den, dens_L, dens_R, dens_U, dens_D, dens_F, dens_B, GRIDS_X, GRIDS_Y, GRIDS_Z );
				kernelHandleHalo __device_func__ ( dev_u,   velu_L, velu_R, velu_U, velu_D, velu_F, velu_B, GRIDS_X, GRIDS_Y, GRIDS_Z );
				kernelHandleHalo __device_func__ ( dev_v,   velv_L, velv_R, velv_U, velv_D, velv_F, velv_B, GRIDS_X, GRIDS_Y, GRIDS_Z );
				kernelHandleHalo __device_func__ ( dev_w,   velw_L, velw_R, velw_U, velw_D, velw_F, velw_B, GRIDS_X, GRIDS_Y, GRIDS_Z );

				SolveNavierStokesEquation( DELTATIME/2.f, false );

				ExitBullet(i,j,k);

				gpu_node[Index(i,j,k,GNODES_X)]->updated = true;
			}
		}

		WriteBuffers();
	}

	double rate = 1.f/(double)HNODES_X;

	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{
		ptr = host_node[i];
		kernelInterLeafGrids __device_func__ ( gd_density, dev_density[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterLeafGrids __device_func__ ( gd_velocity_u, dev_velocity_u[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterLeafGrids __device_func__ ( gd_velocity_v, dev_velocity_v[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelInterLeafGrids __device_func__ ( gd_velocity_w, dev_velocity_w[i], ptr->x, ptr->y, ptr->z, rate, GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	kernelClearHalo __device_func__ ( gd_density, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelClearHalo __device_func__ ( gd_velocity_u, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelClearHalo __device_func__ ( gd_velocity_v, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelClearHalo __device_func__ ( gd_velocity_w, GRIDS_X, GRIDS_Y, GRIDS_Z );
};

void FluidSimProc::AddSource( void )
{
	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( decrease_times eqt 0 )
	{
		kernelAddSource __device_func__ ( dev_den, dev_u, dev_v, dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z );

		if ( m_scHelper.GetCUDALastError( "device kernel: kernelPickData failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}

		increase_times++;

		if ( increase_times eqt 200 )
		{
			decrease_times = increase_times;
			increase_times = 0;
		}
	}
	else
	{
		decrease_times--;
	}
};

void FluidSimProc::SolveNavierStokesEquation( cdouble timestep, bool add )
{
	if ( add ) AddSource();
	VelocitySolver( timestep );
	DensitySolver( timestep );
};

void FluidSimProc::VelocitySolver( cdouble timestep )
{
	// diffuse the velocity field (per axis):
	Diffusion( dev_u0, dev_u, VISOCITY );
	Diffusion( dev_v0, dev_v, VISOCITY );
	Diffusion( dev_w0, dev_w, VISOCITY );

	if ( m_scHelper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );

	if ( m_scHelper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	// advect the velocity field (per axis):
	Advection( dev_u0, dev_u, timestep, dev_u, dev_v, dev_w );
	Advection( dev_v0, dev_v, timestep, dev_u, dev_v, dev_w );
	Advection( dev_w0, dev_w, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolver( cdouble timestep )
{
	Diffusion( dev_den0, dev_den, DIFFUSION );
	std::swap( dev_den0, dev_den );
	Advection ( dev_den, dev_den0, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	for ( int k=0; k<20; k++)
	{
		kernelJacobi<<<gridDim,blockDim>>>( out, in, diff, divisor, GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};

void FluidSimProc::Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelGridAdvection<<<gridDim,blockDim>>>( out, in, timestep, u, v, w, GRIDS_X, GRIDS_Y, GRIDS_Z );
};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_Y * GRIDS_Z;
	Jacobi ( out, in, rate, 1+6*rate );
};

void FluidSimProc::Projection( double *u, double *v, double *w, double *div, double *p )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// the velocity gradient
	kernelGradient<<<gridDim,blockDim>>>( div, p, u, v, w, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract<<<gridDim,blockDim>>>( u, v, w, p, GRIDS_X, GRIDS_Y, GRIDS_Z );
};

void FluidSimProc::ReadBuffers( void )
{
	int ni, nj, nk;
	ni = nj = nk = 0;

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k = 0; k < GNODES_X; k++ ) for ( int j = 0; j < GNODES_X; j++ ) for ( int i = 0; i < GNODES_X; i++ )
	{
		ni = m_cursor.x + i;
		nj = m_cursor.y + j;
		nk = m_cursor.z + k;

		/* load node status */
		gpu_node[Index(i,j,k,GNODES_X)]->x = host_node[Index(ni,nj,nk,HNODES_X)]->x;
		gpu_node[Index(i,j,k,GNODES_X)]->y = host_node[Index(ni,nj,nk,HNODES_X)]->y;
		gpu_node[Index(i,j,k,GNODES_X)]->z = host_node[Index(ni,nj,nk,HNODES_X)]->z;
		gpu_node[Index(i,j,k,GNODES_X)]->updated  = host_node[Index(ni,nj,nk,HNODES_X)]->updated;
		gpu_node[Index(i,j,k,GNODES_X)]->active  = host_node[Index(ni,nj,nk,HNODES_X)]->active;

		/* load data */
		kernelCopyBuffers __device_func__ ( node_density[Index(i,j,k,GNODES_X)], dev_density[Index(ni,nj,nk,HNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( node_velocity_u[Index(i,j,k,GNODES_X)], dev_velocity_u[Index(ni,nj,nk,HNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( node_velocity_v[Index(i,j,k,GNODES_X)], dev_velocity_v[Index(ni,nj,nk,HNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( node_velocity_w[Index(i,j,k,GNODES_X)], dev_velocity_w[Index(ni,nj,nk,HNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( node_obstacle[Index(i,j,k,GNODES_X)],   dev_obstacle[Index(ni,nj,nk,HNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );

	}
};

void FluidSimProc::WriteBuffers( void )
{
	int ni, nj, nk; ni = nj = nk = 0;

	//	m_scHelper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	for ( int k = 0; k < GNODES_X; k++ ) for ( int j = 0; j < GNODES_X; j++ ) for ( int i = 0; i < GNODES_X; i++ )
	{
		ni = m_cursor.x + i;
		nj = m_cursor.y + j;
		nk = m_cursor.z + k;

		/* updated the node status */
		host_node[Index(ni,nj,nk,HNODES_X)]->updated = gpu_node[Index(i,j,k,GNODES_X)];

		/* updating the data */
		kernelCopyBuffers __device_func__ ( dev_density[Index(ni,nj,nk,HNODES_X)], node_density[Index(i,j,k,GNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dev_velocity_u[Index(ni,nj,nk,HNODES_X)], node_velocity_u[Index(i,j,k,GNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dev_velocity_v[Index(ni,nj,nk,HNODES_X)], node_velocity_v[Index(i,j,k,GNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelCopyBuffers __device_func__ ( dev_velocity_w[Index(ni,nj,nk,HNODES_X)], node_velocity_w[Index(i,j,k,GNODES_X)], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};  
#endif // 0
