/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 03, 2014
* <File Name>     FluidSimProcKernels.cu
*/

#include <iostream>
#include <utility>
#include "MacroDefinition.h"
#include "FrameworkDynamic.h"
#include "Kernels.h"

using namespace sge;

FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	if ( !AllocateResource ( fluid ) ) { FreeResource (); exit (1); }

	/* build order */
	CreateTopology();
	
	/* clear buffer */
	ZeroBuffers();

	/* set boundary */
	InitBoundary( 1, 0, 1 );

	/* 上传节点数据 */
	UploadNodes();

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

	m_node_size = GRIDS_X * GRIDS_X * GRIDS_X * sizeof(double);
	m_volm_size = VOLUME_X * VOLUME_X * VOLUME_X * sizeof(SGUCHAR);

	increase_times = decrease_times = 0;
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
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrLeft  = host_node[cudaIndex3D( i-1, j, k, NODES_X )];
				/* right */
				if ( i <= NODES_X - 2 )
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrRight = host_node[cudaIndex3D( i+1, j, k, NODES_X )];
				/* down */
				if ( j >= 1 )
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrDown  = host_node[cudaIndex3D( i, j-1, k, NODES_X )];
				/* up */
				if ( j <= NODES_X - 2 )
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrUp    = host_node[cudaIndex3D( i, j+1, k, NODES_X )];
				/* back */
				if ( k >= 1 )
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrBack  = host_node[cudaIndex3D( i, j, k-1, NODES_X )];
				/* front */
				if ( k <= NODES_X - 2 )
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrFront = host_node[cudaIndex3D( i, j, k+1, NODES_X )];

				host_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.x = i;
				host_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.y = j;
				host_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.z = k;
			}
		}
	}
};

/* 上传内存节点数据 */
void FluidSimProc::UploadNodes( void )
{
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		cudaMemcpy( dev_density_s[i],    host_density[i],    m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velocity_u_s[i], host_velocity_u[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velocity_v_s[i], host_velocity_v[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velocity_w_s[i], host_velocity_w[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_obstacle[i],     host_obstacle[i],   m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed when upload nodes to device", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit(1);
		}
	}
};

/* 下载缓存节点数据 */
void FluidSimProc::DownloadNodes( void )
{
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		cudaMemcpy( host_density[i],    dev_density_t[i],    m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_u[i], dev_velocity_u_t[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_velocity_v_t[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_velocity_w_t[i], m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed when download nodes to host", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit(1);
		}
	}
};


bool FluidSimProc::AllocateResource ( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
	{
		helper.GetCUDALastError ( "cudaSetDevices", __FILE__, __LINE__ );
		return false;
	}

	/* 创建流体数据节点 */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		host_density.push_back( ptrDens );
		host_velocity_u.push_back( ptrU );
		host_velocity_v.push_back( ptrV );
		host_velocity_w.push_back( ptrW );
		host_obstacle.push_back( ptrObs );

		/* 创建拓扑结构节点 */
		SimNode *node = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft = node->ptrRight = nullptr;
		node->ptrDown = node->ptrUp = nullptr;
		
		host_node.push_back( node );
	}

	/* 创建GPU节点 STEP 01 */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		dev_density_s.push_back( ptrDens );
		dev_velocity_u_s.push_back( ptrU );
		dev_velocity_v_s.push_back( ptrV );
		dev_velocity_w_s.push_back( ptrW );

		dev_obstacle.push_back( ptrObs );
	}

	/* 创建GPU节点 STEP 02 */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW;

		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		dev_density_t.push_back( ptrDens );
		dev_velocity_u_t.push_back( ptrU );
		dev_velocity_v_t.push_back( ptrV );
		dev_velocity_w_t.push_back( ptrW );
	}

	/* allocate memory on GPU devices */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK )
			return false;

		dev_buffers.push_back(ptr);
	}

	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )  return false;

	/* finally */
	return true;
}  

void FluidSimProc::FreeResource ( void )
{
	/* free host resource */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		helper.FreeHostBuffers( 1, &host_density[i] );
		helper.FreeHostBuffers( 1, &host_velocity_u[i] );
		helper.FreeHostBuffers( 1, &host_velocity_v[i] );
		helper.FreeHostBuffers( 1, &host_velocity_w[i] );
		helper.FreeHostBuffers( 1, &host_obstacle[i] );

		helper.FreeDeviceBuffers( 1, &dev_obstacle[i] );

		helper.FreeDeviceBuffers( 1, &dev_density_t[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_u_t[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_v_t[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_w_t[i] );
		
		helper.FreeDeviceBuffers( 1, &dev_density_s[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_u_s[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_v_s[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_w_s[i] );	
	}

	/* free device resource */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		helper.FreeDeviceBuffers( 1, &dev_buffers[i] );
	}
}

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				/* for fluid simulation, copy the data to device */
				LoadNode(i,j,k);
					
				/* Fluid process */
				AddSource();
				VelocitySolver();
				DensitySolver();
					
				/* tracing */
				TracingTheFlow();

				/* retrieve data back to host */
				SaveNode(i,j,k);

				if ( cudaThreadSynchronize() not_eq cudaSuccess )
				{
					printf( "cudaThreadSynchronize failed\n" );
					FreeResource();
					exit( 1 );
				}
			}
		}
	}

	/* finally, generate volumetric image */
	GetVolumetric( fluid );
};

void FluidSimProc::GetVolumetric( FLUIDSPARAM *fluid )
{
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::LoadNode( int i, int j, int k )
{
	SimNode *ptr = host_node[cudaIndex3D( i, j, k, NODES_X )];

	/* upload center node to GPU device */
	cudaMemcpy( dev_u, host_velocity_u[cudaIndex3D( i, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_v, host_velocity_v[cudaIndex3D( i, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_w, host_velocity_w[cudaIndex3D( i, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_den,  host_density[cudaIndex3D( i, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_obs, host_obstacle[cudaIndex3D( i, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

	if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	/* upload neighbouring buffers to GPU device */
	if ( ptr->ptrLeft not_eq nullptr )
	{
		cudaMemcpy( velu_L, host_velocity_u[cudaIndex3D( i-1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velv_L, host_velocity_v[cudaIndex3D( i-1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velw_L, host_velocity_w[cudaIndex3D( i-1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dens_L,    host_density[cudaIndex3D( i-1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrRight not_eq nullptr )
	{
		cudaMemcpy( velu_R, host_velocity_u[cudaIndex3D( i+1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velv_R, host_velocity_v[cudaIndex3D( i+1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velw_R, host_velocity_w[cudaIndex3D( i+1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dens_R,    host_density[cudaIndex3D( i+1, j, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrUp not_eq nullptr )
	{
		cudaMemcpy( velu_U, host_velocity_u[cudaIndex3D( i, j+1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velv_U, host_velocity_v[cudaIndex3D( i, j+1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velw_U, host_velocity_w[cudaIndex3D( i, j+1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dens_U,    host_density[cudaIndex3D( i, j+1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrDown not_eq nullptr )
	{
		cudaMemcpy( velu_D, host_velocity_u[cudaIndex3D( i, j-1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velv_D, host_velocity_v[cudaIndex3D( i, j-1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velw_D, host_velocity_w[cudaIndex3D( i, j-1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dens_D,    host_density[cudaIndex3D( i, j-1, k, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrFront not_eq nullptr )
	{
		cudaMemcpy( velu_F, host_velocity_u[cudaIndex3D( i, j, k+1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velv_F, host_velocity_v[cudaIndex3D( i, j, k+1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velw_F, host_velocity_w[cudaIndex3D( i, j, k+1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dens_F,    host_density[cudaIndex3D( i, j, k+1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrBack not_eq nullptr )
	{
		cudaMemcpy( velu_B, host_velocity_u[cudaIndex3D( i, j, k-1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velv_B, host_velocity_v[cudaIndex3D( i, j, k-1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( velw_B, host_velocity_w[cudaIndex3D( i, j, k-1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dens_B,    host_density[cudaIndex3D( i, j, k-1, NODES_X )], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
};

void FluidSimProc::SaveNode( int i, int j, int k )
{
	SimNode *ptr = host_node[cudaIndex3D( i, j, k, NODES_X )];

	/* draw data back */
	cudaMemcpy( host_velocity_u[cudaIndex3D( i, j, k, NODES_X )], dev_u, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_v[cudaIndex3D( i, j, k, NODES_X )], dev_v, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_w[cudaIndex3D( i, j, k, NODES_X )], dev_w, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_density[cudaIndex3D( i, j, k, NODES_X )],  dev_den, m_node_size, cudaMemcpyDeviceToHost );

	if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	/* draw neighbouring buffers back */
	if ( ptr->ptrLeft not_eq nullptr )
	{
		cudaMemcpy( host_velocity_u[cudaIndex3D( i-1, j, k, NODES_X )], velu_L, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[cudaIndex3D( i-1, j, k, NODES_X )], velv_L, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[cudaIndex3D( i-1, j, k, NODES_X )], velw_L, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy(    host_density[cudaIndex3D( i-1, j, k, NODES_X )], dens_L, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrRight not_eq nullptr )
	{
		cudaMemcpy( host_velocity_u[cudaIndex3D( i+1, j, k, NODES_X )], velu_R, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[cudaIndex3D( i+1, j, k, NODES_X )], velv_R, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[cudaIndex3D( i+1, j, k, NODES_X )], velw_R, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy(    host_density[cudaIndex3D( i+1, j, k, NODES_X )], dens_R, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrUp not_eq nullptr )
	{
		cudaMemcpy( host_velocity_u[cudaIndex3D( i, j+1, k, NODES_X )], velu_U, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[cudaIndex3D( i, j+1, k, NODES_X )], velv_U, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[cudaIndex3D( i, j+1, k, NODES_X )], velw_U, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy(    host_density[cudaIndex3D( i, j+1, k, NODES_X )], dens_U, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrDown not_eq nullptr )
	{
		cudaMemcpy( host_velocity_u[cudaIndex3D( i, j-1, k, NODES_X )], velu_D, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[cudaIndex3D( i, j-1, k, NODES_X )], velv_D, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[cudaIndex3D( i, j-1, k, NODES_X )], velw_D, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy(    host_density[cudaIndex3D( i, j-1, k, NODES_X )], dens_D, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrFront not_eq nullptr )
	{
		cudaMemcpy( host_velocity_u[cudaIndex3D( i, j, k+1, NODES_X )], velu_F, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[cudaIndex3D( i, j, k+1, NODES_X )], velv_F, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[cudaIndex3D( i, j, k+1, NODES_X )], velw_F, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy(    host_density[cudaIndex3D( i, j, k+1, NODES_X )], dens_F, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrBack not_eq nullptr )
	{
		cudaMemcpy( host_velocity_u[cudaIndex3D( i, j, k-1, NODES_X )], velu_B, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[cudaIndex3D( i, j, k-1, NODES_X )], velv_B, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[cudaIndex3D( i, j, k-1, NODES_X )], velw_B, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy(    host_density[cudaIndex3D( i, j, k-1, NODES_X )], dens_B, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	/* draw volumetric data back */
	cudaDeviceDim3D();
	kernelPickData <<<gridDim, blockDim>>>( dev_visual, dev_den, i * GRIDS_X, j * GRIDS_X, k * GRIDS_X );

};

void FluidSimProc::AddSource( void )
{
	if ( decrease_times eqt 0 )
	{
		cudaDeviceDim3D();
		kernelAddSource<<<gridDim, blockDim>>> ( dev_den, dev_u, dev_v, dev_w, dev_obs );
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

void FluidSimProc::InitBoundary( int i, int j, int k )
{
	cudaDeviceDim3D();

	/* set boundary */
	kernelZeroGrids<<<gridDim, blockDim>>>( dev_obs );
	kernelSetBoundary<<<gridDim, blockDim>>>( dev_obs );
	
	/* 将边界条件拷贝至内存 */
	if ( cudaMemcpy( host_obstacle[cudaIndex3D( i,j,k,NODES_X )], dev_obs,
		m_node_size, cudaMemcpyDeviceToHost) not_eq cudaSuccess )
	{
		helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::VelocitySolver( void )
{
	// diffuse the velocity field (per axis):
	hostDiffusion( dev_u0, dev_u, VISOCITY, dev_obs, MACRO_VELOCITY_U );
	hostDiffusion( dev_v0, dev_v, VISOCITY, dev_obs, MACRO_VELOCITY_V );
	hostDiffusion( dev_w0, dev_w, VISOCITY, dev_obs, MACRO_VELOCITY_W );
	
	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );
	
	// advect the velocity field (per axis):
	hostAdvection( dev_u0, dev_u, dev_obs, MACRO_VELOCITY_U, dev_u, dev_v, dev_w );
	hostAdvection( dev_v0, dev_v, dev_obs, MACRO_VELOCITY_V, dev_u, dev_v, dev_w );
	hostAdvection( dev_w0, dev_w, dev_obs, MACRO_VELOCITY_W, dev_u, dev_v, dev_w );
	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );
};

void FluidSimProc::DensitySolver( void )
{
	hostDiffusion( dev_den0, dev_den, DIFFUSION, dev_obs, MACRO_DENSITY );
	std::swap( dev_den0, dev_den );
	hostAdvection ( dev_den, dev_den0, dev_obs, MACRO_DENSITY, dev_u, dev_v, dev_w );
};

void FluidSimProc::ZeroBuffers( void )
{
	cudaDeviceDim3D();

	/* zero GPU buffer */
	for ( int i = 0; i < dev_buffers_num; i++ )
		kernelZeroGrids <<<gridDim, blockDim>>> ( dev_buffers[i] );

	/* zero host buffer */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_density_s[i] );
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_velocity_u_s[i] );
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_velocity_v_s[i] );
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_velocity_w_s[i] );

		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_density_t[i] );
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_velocity_u_t[i] );
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_velocity_v_t[i] );
		kernelZeroGrids<<<gridDim,blockDim>>> ( dev_velocity_w_t[i] );
	}

	/* zero visual buffer */
	kernelZeroVolumetric <<< gridDim, blockDim>>> ( dev_visual );
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );

	DownloadNodes();
};

void FluidSimProc::TracingTheFlow( void )
{

};