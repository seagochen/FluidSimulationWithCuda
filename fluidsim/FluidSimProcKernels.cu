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

//					hostAddSource( dev_center, SG_VELOCITY_U_FIELD );
//					hostAddSource( dev_center, SG_VELOCITY_V_FIELD );
//					hostAddSource( dev_center, SG_VELOCITY_W_FIELD );
//					hostAddSource( dev_center, SG_DENSITY_FIELD );


					/* 计算结束 */
					DownloadBuffers();
					hostPickData( dev_L0_visual, dev_center, &nodeIX ); 
				}
			}
		}
	}
};

/* allocate resource */
bool FluidSimProc::AllocateResource( void )
{
	/* allocate device L-0 buffers */
	for ( int i = 0; i < dev_L0_vector_num; i++ )
	{
		SGSIMPLENODES *buf;
		if ( m_helper.CreateCUDABuffers( &buf ) not_eq SG_RUNTIME_OK )
		{
			m_helper.CheckRuntimeErrors( "cudaMalloc failed", __FILE__, __LINE__ );
			return false;
		}
		dev_L0_vector.push_back( buf );
	}

	/* allocate device L-1 buffer */
	if ( m_helper.CreateCUDABuffers( &dev_L1_bufs ) not_eq SG_RUNTIME_OK )
	{
		m_helper.CheckRuntimeErrors( "cudaMalloc failed", __FILE__, __LINE__ );
		return false;
	}

	/* allocate device L-2 buffers */
	for ( int i = 0; i < dev_L2_vector_num; i++ )
	{
		SGSTDGRID *buf;
		if ( m_helper.CreateCUDABuffers( &buf ) not_eq SG_RUNTIME_OK )
		{
			m_helper.CheckRuntimeErrors( "cudaMalloc failed", __FILE__, __LINE__ );
			return false;
		}
		dev_L2_vector.push_back( buf );
	}

	/* allocate host L-0 buffers */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++)
	{
		SGHOSTNODE *buf;
		if ( m_helper.CreateHostBuffers( buf ) not_eq SG_RUNTIME_OK )
		{
			printf( "malloc falied\n" );
			return false;
		}
		host_L0_vector.push_back( buf );
	}

	/* allocate visual buffers */	
	if ( m_helper.CreateVolumetricBuffers( &host_L0_visual, &dev_L0_visual ) not_eq SG_RUNTIME_OK )
	{
		m_helper.CheckRuntimeErrors( "cudaMalloc failed", __FILE__, __LINE__ );
		return false;
	}

	/* allocate temporary buffers */
	if ( m_helper.CreateDoubleBuffers( TPBUFFER_X, 1, &dev_L0_temps ) not_eq SG_RUNTIME_OK )
	{
		printf( "malloc falied\n" );
		return false;
	}
		
	return true;
};

/* when program existed, release resource */
void FluidSimProc::FreeResource( void )
{
	/* free device L-0 buffers */
	for ( int i = 0; i < dev_L0_vector_num; i++ )
	{
		cudaFree( dev_L0_vector[i] );
	}

	/* free device L-1 buffers */
	cudaFree( dev_L1_bufs );

	/* free device L-2 buffers */
	for ( int i = 0; i < dev_L2_vector_num; i++ )
	{
		cudaFree( dev_L2_vector[i] );
	}

	/* free host L-0 buffers */
	for ( int i = 0; i < host_L0_vector.size(); i++ )
	{
		SAFE_FREE_PTR( host_L0_vector[i] );
	}

	/* free L-0 visual buffers */
	SAFE_FREE_PTR( host_L0_visual );
	cudaFree( dev_L0_visual );
	cudaFree( dev_L0_temps );
};

/* zero the buffers for fluid simulation */
void FluidSimProc::ZeroBuffers( void )
{
	/* zero center node first */
	hostZeroBuffer( dev_center );

	size_t size = GRIDS_X * GRIDS_X * GRIDS_X;

	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		if ( cudaMemcpy( host_L0_vector[i]->ptrGrids, dev_center, 
			sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) != cudaSuccess )
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
		host_L0_vector[ix]->bActive = true;

		/* print status */
		if ( host_L0_vector[ix]->bActive )			
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
		host_L0_vector[ix]->bActive = false;

		/* print status */
		if ( host_L0_vector[ix]->bActive )			
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
	int IX = 0;

	for ( int i = 0; i < GRIDS_X; i++ )
	{
		for ( int j = 0; j < GRIDS_X; j++ )
		{
			for ( int k = 0; k < GRIDS_X; k++ )
			{
				for ( IX = 0; IX < host_L0_vector.size(); IX++ )
				{
					host_L0_vector[IX]->ptrGrids[cudaIndex3D(i,j,k,NODES_X)].obstacle = SG_BLANK;
				}
			}
		}
	}

	IX = cudaIndex3D(1,1,0,NODES_X);

	int half = GRIDS_X / 2;
//	host_L0_vector[IX]->ptrGrids[cudaIndex3D(half,half,half,GRIDS_X)].obstacle = SG_SOURCE;
//	host_L0_vector[IX]->ptrGrids[cudaIndex3D(half-1,half,0,NODES_X)].obstacle = SG_SOURCE;
//	host_L0_vector[IX]->ptrGrids[cudaIndex3D(half+1,half,0,NODES_X)].obstacle = SG_SOURCE;
//	host_L0_vector[IX]->ptrGrids[cudaIndex3D(half,half-1,0,NODES_X)].obstacle = SG_SOURCE;
//	host_L0_vector[IX]->ptrGrids[cudaIndex3D(half,half+1,0,NODES_X)].obstacle = SG_SOURCE;
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

				if ( index >= host_L0_vector.size() or index < 0 )
				{
					printf ( "index out of range! %s, line: %d \n", __FILE__, __LINE__ );
					exit ( 1 );
				}

				/* left */
				if ( i >= 1 )
					host_L0_vector[index]->ptrLeft = host_L0_vector[index-1];
				/* right */
				if ( i <= NODES_X - 2 )
					host_L0_vector[index]->ptrRight = host_L0_vector[index+1];
				/* down */
				if ( j >= 1 )
					host_L0_vector[index]->ptrDown = host_L0_vector[index-NODES_X];
				/* up */
				if ( j <= NODES_X - 2 )
					host_L0_vector[index]->ptrUp = host_L0_vector[index+NODES_X];
				/* back */
				if ( k >= 1 )
					host_L0_vector[index]->ptrBack = host_L0_vector[index-NODES_X*NODES_X];
				/* front */
				if ( k <= NODES_X - 2 )
					host_L0_vector[index]->ptrFront = host_L0_vector[index+NODES_X*NODES_X];

				host_L0_vector[index]->n3Pos.x = i;
				host_L0_vector[index]->n3Pos.y = j;
				host_L0_vector[index]->n3Pos.z = k;

				printf ( "no: %d | offset: %d%d%d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index,
					host_L0_vector[index]->n3Pos.x, 
					host_L0_vector[index]->n3Pos.y, 
					host_L0_vector[index]->n3Pos.z,
					host_L0_vector[index]->ptrLeft != NULL,
					host_L0_vector[index]->ptrRight != NULL,
					host_L0_vector[index]->ptrUp != NULL,
					host_L0_vector[index]->ptrDown != NULL,
					host_L0_vector[index]->ptrFront != NULL,
					host_L0_vector[index]->ptrBack != NULL );
			}
		}
	}

	printf( "-----------------------------------------------\n" );
};

/* copy host data to CUDA device */
void FluidSimProc::UploadBuffers( void )
{
	int index = cudaIndex3D( nodeIX.x, nodeIX.y, nodeIX.z, NODES_X );

	/* zero all buffers first */
	hostZeroBuffer( dev_center );
	hostZeroBuffer( dev_left );
	hostZeroBuffer( dev_right );
	hostZeroBuffer( dev_up );
	hostZeroBuffer( dev_down );
	hostZeroBuffer( dev_front );
	hostZeroBuffer( dev_back );
	//printf( "buffers cleared!\n" );

	SGHOSTNODE *ptr = host_L0_vector[index];
	size_t size = sizeof( SGSTDGRID ) * GRIDS_X * GRIDS_X * GRIDS_X;

	if ( ptr->ptrLeft not_eq nullptr and 
		cudaMemcpy( dev_left, ptr->ptrLeft->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrRight not_eq nullptr and 
		cudaMemcpy( dev_right, ptr->ptrRight->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrUp not_eq nullptr and 
		cudaMemcpy( dev_up, ptr->ptrUp->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrDown not_eq nullptr and 
		cudaMemcpy( dev_down, ptr->ptrDown->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrFront not_eq nullptr and 
		cudaMemcpy( dev_front, ptr->ptrFront->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrBack not_eq nullptr and 
		cudaMemcpy( dev_back, ptr->ptrBack->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( dev_center, ptr->ptrGrids, size, cudaMemcpyHostToDevice ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	//printf( "data has been uploaded to CUDA device!\n" );
};

/* retrieve data back to host */
void FluidSimProc::DownloadBuffers( void )
{
	int index = cudaIndex3D( nodeIX.x, nodeIX.y, nodeIX.z, NODES_X );

	SGHOSTNODE *ptr = host_L0_vector[index];
	size_t size = sizeof( SGSTDGRID ) * GRIDS_X * GRIDS_X * GRIDS_X;

	if ( ptr->ptrLeft not_eq nullptr and 
		cudaMemcpy( ptr->ptrLeft->ptrGrids, dev_left, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrRight not_eq nullptr and 
		cudaMemcpy( ptr->ptrRight->ptrGrids, dev_right, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrUp not_eq nullptr and 
		cudaMemcpy( ptr->ptrUp->ptrGrids, dev_up, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrDown not_eq nullptr and 
		cudaMemcpy( ptr->ptrDown->ptrGrids, dev_down, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrFront not_eq nullptr and 
		cudaMemcpy( ptr->ptrFront->ptrGrids, dev_front, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( ptr->ptrBack not_eq nullptr and 
		cudaMemcpy( ptr->ptrBack->ptrGrids, dev_back, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	if ( cudaMemcpy( ptr->ptrGrids, dev_center, size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}
	//printf( "data has been downloaded to host!\n" );
};

/* retrieve the density back and load into volumetric data for rendering */
void FluidSimProc::PickVolumetric( FLUIDSPARAM *fluid )
{
	if ( cudaMemcpy( host_L0_visual, dev_L0_visual,
		sizeof(SGUCHAR) * VOLUME_X * VOLUME_X * VOLUME_X, cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

	fluid->volume.ptrData = host_L0_visual;
};