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

void FluidSimProc::freeHostRes( void )
{
	/* free node resource */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		/* release host resource */
		helper.FreeHostBuffers( 1, &host_density[i] );
		helper.FreeHostBuffers( 1, &host_velocity_u[i] );
		helper.FreeHostBuffers( 1, &host_velocity_v[i] );
		helper.FreeHostBuffers( 1, &host_velocity_w[i] );
		helper.FreeHostBuffers( 1, &host_obstacle[i] );

		helper.FreeDeviceBuffers( 1, &dev_density[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_u[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_v[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_w[i] );
		helper.FreeDeviceBuffers( 1, &dev_obstacle[i] );
	}
};

void FluidSimProc::freeDeviceRes( void )
{
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{
		/* release device resource */
		helper.FreeDeviceBuffers( 1, &node_obstacle[i] );
		helper.FreeDeviceBuffers( 1, &node_density[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_u[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_v[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_w[i] );	
	}

	helper.FreeDeviceBuffers( 5, &gd_density, &gd_velocity_u, &gd_velocity_v, &gd_velocity_w, &gd_obstacle );
};

void FluidSimProc::freeShareBuffers( void )
{
	for ( int i = 0; i < dev_buffers_num; i++ ) 
		helper.FreeDeviceBuffers( 1, &dev_buffers[i] );
};

void FluidSimProc::freeVisualBuffers( void )
{
	helper.FreeDeviceBuffers( 1, &dev_visual );
	helper.FreeHostBuffers( 1, &host_visual );
};

void FluidSimProc::FreeResource ( void )
{
	freeHostRes();
	freeDeviceRes();	
	freeShareBuffers();
	freeVisualBuffers();
}




void FluidSimProc::zeroTempoBuffers( void )
{
	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int i = 0; i < dev_buffers_num; i++ )
//		kernelZeroBuffers __device_func__ ( dev_buffers[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroGrids  __device_func__ ( dev_buffers[i] );
};

void FluidSimProc::zeroGlobalNode( void )
{
	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	kernelZeroGrids __device_func__ ( gd_density );
	kernelZeroGrids __device_func__ ( gd_velocity_u );
	kernelZeroGrids __device_func__ ( gd_velocity_v );
	kernelZeroGrids __device_func__ ( gd_velocity_w );
	 
//	 kernelZeroBuffers __device_func__ ( gd_density, GRIDS_X, GRIDS_Y, GRIDS_Z );
//	 kernelZeroBuffers __device_func__ ( gd_velocity_u, GRIDS_X, GRIDS_Y, GRIDS_Z );
//	 kernelZeroBuffers __device_func__ ( gd_velocity_v, GRIDS_X, GRIDS_Y, GRIDS_Z );
//	 kernelZeroBuffers __device_func__ ( gd_velocity_w, GRIDS_X, GRIDS_Y, GRIDS_Z );
};

void FluidSimProc::zeroDeivceRes( void )
{
	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{
		kernelZeroGrids __device_func__ ( node_density[i] );
		kernelZeroGrids __device_func__ ( node_velocity_u[i] );
		kernelZeroGrids __device_func__ ( node_velocity_v[i] );
		kernelZeroGrids __device_func__ ( node_velocity_w[i] );

//		kernelZeroBuffers __device_func__ ( node_density[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
//		kernelZeroBuffers __device_func__ ( node_velocity_u[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
//		kernelZeroBuffers __device_func__ ( node_velocity_v[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
//		kernelZeroBuffers __device_func__ ( node_velocity_w[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};

void FluidSimProc::zeroHostRes( void )
{
	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
//		kernelZeroBuffers __device_func__ ( dev_density[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
//		kernelZeroBuffers __device_func__ ( dev_velocity_u[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
//		kernelZeroBuffers __device_func__ ( dev_velocity_v[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
//		kernelZeroBuffers __device_func__ ( dev_velocity_w[i], GRIDS_X, GRIDS_Y, GRIDS_Z );

		kernelZeroGrids __device_func__ ( dev_density[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_u[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_v[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_w[i] );
	}
};

void FluidSimProc::zeroVisualBuffers( void )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, VOLUME_X, VOLUME_X, VOLUME_X );

	//kernelZeroBuffers __device_func__ ( dev_visual, VOLUME_X, VOLUME_X, VOLUME_X );
};

void FluidSimProc::zeroShareBuffers( void )
{
	helper.DeviceDim1D( &blockDim, &gridDim, THREADS_X, TPBUFFER_X );

//	kernelZeroBuffers __device_func__ ( dev_ntpbuf, TPBUFFER_X );
//	kernelZeroBuffers __device_func__ ( dev_dtpbuf, TPBUFFER_X );
};

void FluidSimProc::ZeroBuffers( void )
{
	zeroTempoBuffers();
	zeroDeivceRes();
	zeroHostRes();
	zeroVisualBuffers();
	zeroShareBuffers();
	zeroGlobalNode();
	
	if ( helper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};





void FluidSimProc::HostToDevice( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
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
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		cudaMemcpy( host_density[i], dev_density[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_u[i], dev_velocity_u[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_velocity_v[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_velocity_w[i], m_node_size, cudaMemcpyDeviceToHost );
	}
};

void FluidSimProc::clearBullet( void )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int i = 11; i < dev_buffers_num; i++ )
{
	//		kernelZeroBuffers __device_func__ ( dev_buffers[i], GRIDS_X, GRIDS_X, GRIDS_X );
	kernelZeroGrids __device_func__ ( dev_buffers[i] );
	}

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickNodeToBullet( int i, int j, int k )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	/* upload center node to GPU device */
	
//	kernelCopyBuffers __device_func__ ( dev_u, node_velocity_u[cudaIndex3D( i, j, k, GNODES_X )], GRIDS_X, GRIDS_X, GRIDS_X );
//	kernelCopyBuffers __device_func__ ( dev_v, node_velocity_v[cudaIndex3D( i, j, k, GNODES_X )], GRIDS_X, GRIDS_X, GRIDS_X );
//	kernelCopyBuffers __device_func__ ( dev_w, node_velocity_w[cudaIndex3D( i, j, k, GNODES_X )], GRIDS_X, GRIDS_X, GRIDS_X );
//	kernelCopyBuffers __device_func__ ( dev_den,  node_density[cudaIndex3D( i, j, k, GNODES_X )], GRIDS_X, GRIDS_X, GRIDS_X );
//	kernelCopyBuffers __device_func__ ( dev_obs, node_obstacle[cudaIndex3D( i, j, k, GNODES_X )], GRIDS_X, GRIDS_X, GRIDS_X );

	kernelCopyGrids __device_func__ ( dev_u, node_velocity_u[cudaIndex3D( i, j, k, GNODES_X )] );
	kernelCopyGrids __device_func__ ( dev_v, node_velocity_v[cudaIndex3D( i, j, k, GNODES_X )] );
	kernelCopyGrids __device_func__ ( dev_w, node_velocity_w[cudaIndex3D( i, j, k, GNODES_X )] );
	kernelCopyGrids __device_func__ ( dev_den,  node_density[cudaIndex3D( i, j, k, GNODES_X )] );
	kernelCopyGrids __device_func__ ( dev_obs, node_obstacle[cudaIndex3D( i, j, k, GNODES_X )] );

	if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickNeighborsToBullet( int i, int j, int k )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	/* upload neighbouring buffers to GPU device */
	ptr = gpu_node[cudaIndex3D( i, j, k, GNODES_X )];
	if ( ptr->ptrLeft not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_L, node_velocity_u[cudaIndex3D( i-1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_L, node_velocity_v[cudaIndex3D( i-1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_L, node_velocity_w[cudaIndex3D( i-1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_L,    node_density[cudaIndex3D( i-1, j, k, GNODES_X )] );
	}

	if ( ptr->ptrRight not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_R, node_velocity_u[cudaIndex3D( i+1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_R, node_velocity_v[cudaIndex3D( i+1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_R, node_velocity_w[cudaIndex3D( i+1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_R,    node_density[cudaIndex3D( i+1, j, k, GNODES_X )] );
	}

	if ( ptr->ptrUp not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_U, node_velocity_u[cudaIndex3D( i, j+1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_U, node_velocity_v[cudaIndex3D( i, j+1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_U, node_velocity_w[cudaIndex3D( i, j+1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_U,    node_density[cudaIndex3D( i, j+1, k, GNODES_X )] );
	}

	if ( ptr->ptrDown not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_D, node_velocity_u[cudaIndex3D( i, j-1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_D, node_velocity_v[cudaIndex3D( i, j-1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_D, node_velocity_w[cudaIndex3D( i, j-1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_D,    node_density[cudaIndex3D( i, j-1, k, GNODES_X )] );
	}

	if ( ptr->ptrFront not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_F, node_velocity_u[cudaIndex3D( i, j, k+1, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_F, node_velocity_v[cudaIndex3D( i, j, k+1, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_F, node_velocity_w[cudaIndex3D( i, j, k+1, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_F,    node_density[cudaIndex3D( i, j, k+1, GNODES_X )] );
	}

	if ( ptr->ptrBack not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_B, node_velocity_u[cudaIndex3D( i, j, k-1, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_B, node_velocity_v[cudaIndex3D( i, j, k-1, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_B, node_velocity_w[cudaIndex3D( i, j, k-1, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_B,    node_density[cudaIndex3D( i, j, k-1, GNODES_X )] );
	}

	if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
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
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	kernelCopyGrids __device_func__( node_velocity_u[cudaIndex3D(i,j,k,GNODES_X)], velu_C );
	kernelCopyGrids __device_func__( node_velocity_v[cudaIndex3D(i,j,k,GNODES_X)], velv_C );
	kernelCopyGrids __device_func__( node_velocity_w[cudaIndex3D(i,j,k,GNODES_X)], velw_C );
	kernelCopyGrids __device_func__(    node_density[cudaIndex3D(i,j,k,GNODES_X)], dens_C );

	if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickImgFromNode( int i, int j, int k )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	i += m_cursor.x;
	j += m_cursor.y;
	k += m_cursor.z;
	kernelPickData __device_func__( dev_visual, dev_den, i, j, k, GRIDS_X );

	if ( helper.GetCUDALastError( "device kernel: kernelPickData failed", __FILE__, __LINE__ ) )
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
//	 kernelZeroBuffers __device_func__ ( dev_obs, GRIDS_X, GRIDS_X, GRIDS_X );
	 kernelZeroGrids __device_func__ ( dev_obs );

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
	
	zeroShareBuffers();
	zeroHostRes();
	SolveRootNode();
//	SolveLeafNode();
	RefreshStatus( fluid );
};

void FluidSimProc::SolveRootNode( void )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );
	zeroTempoBuffers();

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

	/* error is here ! */
//	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
//	{
//		kernelCopyGrids __device_func__ ( dev_den, dev_density[cudaIndex3D(i,j,k,HNODES_X)] );
//		kernelCopyGrids __device_func__ ( dev_u, dev_velocity_u[cudaIndex3D(i,j,k,HNODES_X)] );
//		kernelCopyGrids __device_func__ ( dev_v, dev_velocity_v[cudaIndex3D(i,j,k,HNODES_X)] );
//		kernelCopyGrids __device_func__ ( dev_w, dev_velocity_w[cudaIndex3D(i,j,k,HNODES_X)] );
//
//		kernelClearHalo __device_func__ ( dev_den );
//		kernelClearHalo __device_func__ ( dev_u );
//		kernelClearHalo __device_func__ ( dev_v );
//		kernelClearHalo __device_func__ ( dev_w );
//		
//		SolveNavierStokesEquation( DELTATIME/2.f, false );
//
//		kernelCopyGrids __device_func__ ( dev_density[cudaIndex3D(i,j,k,HNODES_X)], dev_den );
//		kernelCopyGrids __device_func__ ( dev_velocity_u[cudaIndex3D(i,j,k,HNODES_X)], dev_u );
//		kernelCopyGrids __device_func__ ( dev_velocity_v[cudaIndex3D(i,j,k,HNODES_X)], dev_v );
//		kernelCopyGrids __device_func__ ( dev_velocity_w[cudaIndex3D(i,j,k,HNODES_X)], dev_w );
//	}

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
};

void FluidSimProc::SolveLeafNode( void )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );
	zeroTempoBuffers();

#if HNODES_X >= 3

	/* sum density */
	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		kernelSumDensity __device_func__ ( dev_dtpbuf, dev_density[cudaIndex3D(i,j,k,HNODES_X)], cudaIndex3D(i,j,k,HNODES_X) );
	}
	cudaMemcpy( host_dtpbuf, dev_dtpbuf, sizeof(double)*TPBUFFER_X, cudaMemcpyDeviceToHost );

	/* ignore the node with density lower than 1 */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
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
			if ( !gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated
				and gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->active )
#else
			if ( !gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated )
#endif
			{
					LoadBullet(i,j,k);

					kernelHandleHalo __device_func__ ( dev_den, dens_L, dens_R, dens_U, dens_D, dens_F, dens_B );
					kernelHandleHalo __device_func__ ( dev_u,   velu_L, velu_R, velu_U, velu_D, velu_F, velu_B );
					kernelHandleHalo __device_func__ ( dev_v,   velv_L, velv_R, velv_U, velv_D, velv_F, velv_B );
					kernelHandleHalo __device_func__ ( dev_w,   velw_L, velw_R, velw_U, velw_D, velw_F, velw_B );

					SolveNavierStokesEquation( DELTATIME/2.f, false );

					ExitBullet(i,j,k);

					gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated = true;
			}
		}

		WriteBuffers();
	}

	double rate = 1.f/(double)HNODES_X;

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		ptr = host_node[i];
		kernelInterLeafGrids __device_func__ ( gd_density, dev_density[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_u, dev_velocity_u[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_v, dev_velocity_v[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_w, dev_velocity_w[i], ptr->x, ptr->y, ptr->z, rate );
	}

	kernelClearHalo __device_func__ ( gd_density );
	kernelClearHalo __device_func__ ( gd_velocity_u );
	kernelClearHalo __device_func__ ( gd_velocity_v );
	kernelClearHalo __device_func__ ( gd_velocity_w );
};

void FluidSimProc::ReadBuffers( void )
{
	int ni, nj, nk;
	ni = nj = nk = 0;
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int k = 0; k < GNODES_X; k++ ) for ( int j = 0; j < GNODES_X; j++ ) for ( int i = 0; i < GNODES_X; i++ )
	{
		ni = m_cursor.x + i;
		nj = m_cursor.y + j;
		nk = m_cursor.z + k;

		/* load node status */
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->x = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->x;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->y = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->y;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->z = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->z;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated  = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->updated;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->active  = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->active;

		/* load data */
		kernelCopyGrids __device_func__ ( node_density[cudaIndex3D(i,j,k,GNODES_X)], dev_density[cudaIndex3D(ni,nj,nk,HNODES_X)] );
		kernelCopyGrids __device_func__ ( node_velocity_u[cudaIndex3D(i,j,k,GNODES_X)], dev_velocity_u[cudaIndex3D(ni,nj,nk,HNODES_X)] );
		kernelCopyGrids __device_func__ ( node_velocity_v[cudaIndex3D(i,j,k,GNODES_X)], dev_velocity_v[cudaIndex3D(ni,nj,nk,HNODES_X)] );
		kernelCopyGrids __device_func__ ( node_velocity_w[cudaIndex3D(i,j,k,GNODES_X)], dev_velocity_w[cudaIndex3D(ni,nj,nk,HNODES_X)] );
		kernelCopyGrids __device_func__ ( node_obstacle[cudaIndex3D(i,j,k,GNODES_X)],   dev_obstacle[cudaIndex3D(ni,nj,nk,HNODES_X)] );
	}
};

void FluidSimProc::WriteBuffers( void )
{
	int ni, nj, nk; ni = nj = nk = 0;

	for ( int k = 0; k < GNODES_X; k++ ) for ( int j = 0; j < GNODES_X; j++ ) for ( int i = 0; i < GNODES_X; i++ )
	{
		ni = m_cursor.x + i;
		nj = m_cursor.y + j;
		nk = m_cursor.z + k;

		/* updated the node status */
		host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->updated = gpu_node[cudaIndex3D(i,j,k,GNODES_X)];

		/* updating the data */
		kernelCopyGrids __device_func__ ( dev_density[cudaIndex3D(ni,nj,nk,HNODES_X)], node_density[cudaIndex3D(i,j,k,GNODES_X)] );
		kernelCopyGrids __device_func__ ( dev_velocity_u[cudaIndex3D(ni,nj,nk,HNODES_X)], node_velocity_u[cudaIndex3D(i,j,k,GNODES_X)] );
		kernelCopyGrids __device_func__ ( dev_velocity_v[cudaIndex3D(ni,nj,nk,HNODES_X)], node_velocity_v[cudaIndex3D(i,j,k,GNODES_X)] );
		kernelCopyGrids __device_func__ ( dev_velocity_w[cudaIndex3D(ni,nj,nk,HNODES_X)], node_velocity_w[cudaIndex3D(i,j,k,GNODES_X)] );
	}
};