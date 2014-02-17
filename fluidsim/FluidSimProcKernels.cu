/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Nov 15, 2013
* <Last Time>     Feb 17, 2014
* <File Name>     FluidSimProc.cu
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "FluidSimProc.h"
#include "CUDAInterfaces.h"

using namespace sge;

/* 默认的构造函数，分配流体模拟所需要的空间，以及初始化相关参数 */
FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
{
	/* initialize the parameters of fluid simulation */
	InitParams( fluid );

	/* allocate the space for fluid simulation */
	if ( !AllocateResource() )
	{
		FreeResource();
		printf(" malloc buffers for fluid simulation failed! \n");
		exit(1);
	}
	else
		printf( "allocate resource success!\n" );

	/* initialize the nodes */
	InitSimNodes();

	/* building structure order */
	BuildOrder();

	/* select and active a node for fluid simulation */
	ActiveNode( 1, 1, 0 );

	/* zero all buffers */
	ZeroBuffers();
	
	/* finally, print the state message and zero the data */
	printf( "fluid simulation ready...\n" );
};

/* 初始化流体模拟的相关参数 */
void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	/* initilize the status of FPS counter */
	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;
};

/* fluid simulation processing function */
void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( fluid->run )
	{
		for ( int i = 0; i < NODES_X; i++ )
		{
			for ( int j = 0; j < NODES_X; j++ )
			{
				for ( int k = 0; k < NODES_X; k++ )
				{
					// TODO: 在技术升级前，将一直采用简单而明了的轮询法，检查各个计算节点的状态。
					// 在当前的情况下，将所有节点默认为激活状态，这样将可以直观的测试各个节点数据传输情况。
					/* 计算开始 */
					SelectNode( i, j, k );
					UploadBuffers();

					AddSource();
					DensitySolver( dev_dens, dev_vel_u, dev_vel_v, dev_vel_w, dev_dens0 );
					VelocitySolver( dev_vel_u, dev_vel_v, dev_vel_w,
						dev_vel_u0, dev_vel_v0, dev_vel_w0, dev_div, dev_p );
//					hostAddSource( dev_center, SG_VELOCITY_U_FIELD );
//					hostAddSource( dev_center, SG_VELOCITY_V_FIELD );
//					hostAddSource( dev_center, SG_VELOCITY_W_FIELD );
//					hostAddSource( dev_center, SG_DENSITY_FIELD );


					/* 计算结束 */
					DownloadBuffers();
					hostPickData( dev_visual, dev_center, &nodeIX ); 
				}
			}
		}
	}
};

__global__ void kernelAddSource( double *dens, double *u, double *v, double *w, double const *obs )
{
	GetIndex();

	if ( obs[ Index(i,j,k) ] eqt BOUND_SOURCE )
	{
		dens[ Index(i,j,k) ] = SOURCE_DENSITY;
		v[ Index(i,j,k) ] = SOURCE_VELOCITY;
	}
};

/* add source */
void FluidSimProc::AddSource( void )
{
	cudaDeviceDim3D();

	kernelAddSource cudaDevice(gridDim, blockDim)
		( dev_dens, dev_vel_u, dev_vel_v, dev_vel_w, dev_obs );
};

/* allocate resource */
bool FluidSimProc::AllocateResource( void )
{
	size_t size = GRIDS_X * GRIDS_X * GRIDS_X;

	/* allocate device buffers */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *buf;
		if ( m_helper.CreateDeviceBuffers( size, 1, &buf ) not_eq SG_RUNTIME_OK )
			return false;

		dev_buffers.push_back( buf );
	}

	/* allocate host buffers */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++)
	{
		double *dens, *u, *v, *w, *obs;
		if ( m_helper.CreateHostBuffers( size, 5, &dens, &u, &v, &w, &obs ) not_eq SG_RUNTIME_OK )
			return false;
		
		host_density.push_back( dens );
		host_velocity_u.push_back( u );
		host_velocity_v.push_back( v );
		host_velocity_w.push_back( w );
		host_obstacle.push_back( obs );
	}

	/* allocate visual buffers */	
	size = VOLUME_X * VOLUME_X * VOLUME_X;
	if ( m_helper.CreateVolumetricBuffers( size, &host_visual, &dev_visual ) not_eq SG_RUNTIME_OK )
		return false;

	/* push nodes to linker */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		LinkNode* node = (LinkNode*)malloc( sizeof(LinkNode) );
		node->ptrBack = node->ptrDown = node->ptrFront =
			node->ptrLeft = node->ptrRight = node->ptrUp = nullptr;

		if ( node eqt nullptr )
		{
			printf( "malloc node failed\n" );
			return false;
		}

		host_link.push_back( node );
	}	

	return true;
};

/* when program existed, release resource */
void FluidSimProc::FreeResource( void )
{
	/* free device L-0 buffers */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		cudaFree( dev_buffers[i] );
	}

	/* free host L-0 buffers */
	int i;
	for ( i = 0; i < host_density.size(); i++ )
		SAFE_FREE_PTR( host_density[i] );
	for ( i = 0; i < host_velocity_u.size(); i++ )
		SAFE_FREE_PTR( host_velocity_u[i] );
	for ( i = 0; i < host_velocity_v.size(); i++ )
		SAFE_FREE_PTR( host_velocity_v[i] );
	for ( i = 0; i < host_velocity_w.size(); i++ )
		SAFE_FREE_PTR( host_velocity_w[i] );
	for ( i = 0; i < host_obstacle.size(); i++ )
		SAFE_FREE_PTR( host_obstacle[i] );

	/* free L-0 visual buffers */
	SAFE_FREE_PTR( host_visual );
	cudaFree( dev_visual );
};

/* zero the buffers for fluid simulation */
void FluidSimProc::ZeroBuffers( void )
{
	/* zero center node first */
	hostZeroBuffer( dev_center );

	size_t size = GRIDS_X * GRIDS_X * GRIDS_X * sizeof(double);

	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		if ( cudaMemcpy( host_density[i], dev_center, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			exit( 1 );
		}
		if ( cudaMemcpy( host_velocity_u[i], dev_center, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			exit( 1 );
		}
		if ( cudaMemcpy( host_velocity_v[i], dev_center, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			exit( 1 );
		}
		if ( cudaMemcpy( host_velocity_w[i], dev_center, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			exit( 1 );
		}

	}
};

/* choose the node and mark it as actived */
void FluidSimProc::ActiveNode( int i, int j, int k )
{
	int ix = 0;

	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		ix = cudaIndex3D( i, j, k, NODES_X );
		host_link[ix]->active = true;

		/* print status */
		if ( host_link[ix]->active )			
			printf ( "node no.%d is actived!\n", ix );
		else
			printf ( "node no.%d is deactived!\n", ix );
	}	
};

/* choose the node and mark it as deactived */
void FluidSimProc::DeactiveNode( int i, int j, int k )
{
	int ix = 0;

	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		ix = cudaIndex3D( i, j, k, NODES_X );
		host_link[ix]->active = false;

		/* print status */
		if ( host_link[ix]->active )			
			printf ( "node no.%d is actived!\n", ix );
		else
			printf ( "node no.%d is deactived!\n", ix );
	}	
};


void FluidSimProc::SelectNode( int i, int j, int k )
{
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		nodeIX.x = i;
		nodeIX.y = j;
		nodeIX.z = k;
	}
};

/* zero data, set the bounds */
void FluidSimProc::InitSimNodes( void )
{
	hostZeroBuffer( dev_center );

	for ( int i = 0; i < host_obstacle.size(); i++ )
	{
		if ( cudaMemcpy( host_obstacle[i], dev_center, 
			sizeof(double) * GRIDS_X * GRIDS_X * GRIDS_X, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMalloc failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	int half = GRIDS_X / 2;
	double *obs = host_obstacle[0];
	obs[Index(half, half, half)] = BOUND_SOURCE;
	obs[Index(half+1, half, half)] = BOUND_SOURCE;
	obs[Index(half-1, half, half)] = BOUND_SOURCE;
	obs[Index(half, half+1, half)] = BOUND_SOURCE;
	obs[Index(half, half-1, half)] = BOUND_SOURCE;
	obs[Index(half, half, half+1)] = BOUND_SOURCE;
	obs[Index(half, half, half-1)] = BOUND_SOURCE;
};

/* create simulation nodes' topological structure */
void FluidSimProc::BuildOrder( void )
{
	printf( "structure:\n" );
	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				int index = cudaIndex3D( i, j, k, NODES_X );

				if ( index >= host_link.size() or index < 0 )
				{
					printf ( "index out of range! %s, line: %d \n", __FILE__, __LINE__ );
					exit ( 1 );
				}

				/* left */
				if ( i >= 1 )
					host_link[index]->ptrLeft = host_link[index-1];
				/* right */
				if ( i <= NODES_X - 2 )
					host_link[index]->ptrRight = host_link[index+1];
				/* down */
				if ( j >= 1 )
					host_link[index]->ptrDown = host_link[index-NODES_X];
				/* up */
				if ( j <= NODES_X - 2 )
					host_link[index]->ptrUp = host_link[index+NODES_X];
				/* back */
				if ( k >= 1 )
					host_link[index]->ptrBack = host_link[index-NODES_X*NODES_X];
				/* front */
				if ( k <= NODES_X - 2 )
					host_link[index]->ptrFront = host_link[index+NODES_X*NODES_X];

				host_link[index]->n3Pos.x = i;
				host_link[index]->n3Pos.y = j;
				host_link[index]->n3Pos.z = k;

				printf ( "no: %d | offset: %d%d%d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index,
					host_link[index]->n3Pos.x, 
					host_link[index]->n3Pos.y, 
					host_link[index]->n3Pos.z,
					host_link[index]->ptrLeft not_eq nullptr,
					host_link[index]->ptrRight not_eq nullptr,
					host_link[index]->ptrUp not_eq nullptr,
					host_link[index]->ptrDown not_eq nullptr,
					host_link[index]->ptrFront not_eq nullptr,
					host_link[index]->ptrBack not_eq nullptr );
			}
		}
	}

	printf( "-----------------------------------------------\n" );
};

/* copy host data to CUDA device */
void FluidSimProc::UploadBuffers( void )
{
	int ix = cudaIndex3D( nodeIX.x, nodeIX.y, nodeIX.z, NODES_X );

	/* zero all buffers first */
	hostZeroBuffer( dev_dens );
	hostZeroBuffer( dev_vel_u );
	hostZeroBuffer( dev_vel_v );
	hostZeroBuffer( dev_vel_w );

	size_t size = GRIDS_X * GRIDS_X * GRIDS_X * sizeof( double );

	if ( cudaMemcpy( dev_dens, host_density[ix], size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( dev_vel_u, host_velocity_u[ix], size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( dev_vel_v, host_velocity_v[ix], size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( dev_vel_w, host_velocity_w[ix], size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( dev_obs, host_obstacle[ix], size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

};

/* retrieve data back to host */
void FluidSimProc::DownloadBuffers( void )
{
	int ix = cudaIndex3D( nodeIX.x, nodeIX.y, nodeIX.z, NODES_X );

	size_t size = GRIDS_X * GRIDS_X * GRIDS_X * sizeof( double );

	if ( cudaMemcpy( host_density[ix], dev_dens, size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( host_velocity_u[ix], dev_vel_u, size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( host_velocity_v[ix], dev_vel_v, size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( host_velocity_w[ix], dev_vel_w, size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
};

/* retrieve the density back and load into volumetric data for rendering */
void FluidSimProc::PickVolumetric( FLUIDSPARAM *fluid )
{
	if ( cudaMemcpy( host_visual, dev_visual,
		sizeof(SGUCHAR) * VOLUME_X * VOLUME_X * VOLUME_X, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

	fluid->volume.ptrData = host_visual;
};