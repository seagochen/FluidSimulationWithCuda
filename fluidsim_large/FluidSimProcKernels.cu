/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 04, 2014
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
	InitBoundary( 0, 0, 0 );

	/* 上传节点数据 */
	UploadNodes();

	/* finally, print message */
	printf( "fluid simulation ready...\n" );
};

ptrStr FluidSimProc::GetTitleBar( void )
{
	return &m_sz_title; 
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

	m_sz_title = "Excalibur OTL 2.10.00, large-scale. ------------ FPS: %d ";
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
					gpu_node[cudaIndex3D( i, j, k, NODES_X )]->ptrLeft  = gpu_node[cudaIndex3D( i-1, j, k, NODES_X )];
				/* right */
				if ( i <= NODES_X - 2 )
					gpu_node[cudaIndex3D( i, j, k, NODES_X )]->ptrRight = gpu_node[cudaIndex3D( i+1, j, k, NODES_X )];
				/* down */
				if ( j >= 1 )
					gpu_node[cudaIndex3D( i, j, k, NODES_X )]->ptrDown  = gpu_node[cudaIndex3D( i, j-1, k, NODES_X )];
				/* up */
				if ( j <= NODES_X - 2 )
					gpu_node[cudaIndex3D( i, j, k, NODES_X )]->ptrUp    = gpu_node[cudaIndex3D( i, j+1, k, NODES_X )];
				/* back */
				if ( k >= 1 )
					gpu_node[cudaIndex3D( i, j, k, NODES_X )]->ptrBack  = gpu_node[cudaIndex3D( i, j, k-1, NODES_X )];
				/* front */
				if ( k <= NODES_X - 2 )
					gpu_node[cudaIndex3D( i, j, k, NODES_X )]->ptrFront = gpu_node[cudaIndex3D( i, j, k+1, NODES_X )];

				gpu_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.x = i;
				gpu_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.y = j;
				gpu_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.z = k;
			}
		}
	}
};

void FluidSimProc::PrintMSG( void )
{
	using namespace std;

	system( "cls" );
	cout 
		<< "**************** operation to confirm *******************" << endl
		<< "mouse wheel ------------ to rotate the observation matrix" << endl
		<< "keyboard: Q ------------ to quit the program" << endl
		<< "keyboard: Esc ---------- to quit the program" << endl
		<< "keyboard: S ------------ to retrieve the data from GPU" << endl
		<< "keyboard: C ------------ to clear the data of stage" << endl
		<< "**************** fluid simulation info ******************" << endl
		<< "number of GPU nodes for fluid simulation: " << gpu_node.size() << endl
		<< "number of HOST nodes for fluid simulation: " << host_node.size() << endl
		<< "grid size per computation node : 64 x 64 x 64" << endl;
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
	}
};

bool FluidSimProc::AllocateResource ( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess ) return false; 

	/* 创建临时数据 */
	if ( helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(double), 1, &dev_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(int), 1, &dev_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_X*sizeof(double), 1, &host_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_X*sizeof(int), 1, &host_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;

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
	}

	/* 创建拓扑结构节点 */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		gpu_node.push_back( node );
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
	
	/* 更新节点状态 */
	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				/* for fluid simulation, copy the data to device */
				LoadNode(i,j,k) ;

				/* 对节点的状况进行跟踪 */
				InteractNodes(i,j,k);

				/* Fluid process */
				AddSource();
				VelocitySolver();
				DensitySolver();
					
				/* retrieve data back to host */
				SaveNode(i,j,k);
			}
		}
	}

	/* 等待所有GPU kernels运行结束 */
	if ( cudaThreadSynchronize() not_eq cudaSuccess )
	{
		printf( "cudaThreadSynchronize failed\n" );
		FreeResource();
		exit( 1 );
	}

	/* finally, generate volumetric image */
	Finally( fluid );
};

void FluidSimProc::LoadNode( int i, int j, int k )
{
	cudaDeviceDim3D();
	SimNode *ptr = gpu_node[cudaIndex3D( i, j, k, NODES_X )];

	/* upload center node to GPU device */
	kernelCopyGrids __device_func__ ( dev_u, dev_velocity_u_s[cudaIndex3D( i, j, k, NODES_X )] );
	kernelCopyGrids __device_func__ ( dev_v, dev_velocity_v_s[cudaIndex3D( i, j, k, NODES_X )] );
	kernelCopyGrids __device_func__ ( dev_w, dev_velocity_w_s[cudaIndex3D( i, j, k, NODES_X )] );
	kernelCopyGrids __device_func__ ( dev_den,  dev_density_s[cudaIndex3D( i, j, k, NODES_X )] );
	kernelCopyGrids __device_func__ ( dev_obs,   dev_obstacle[cudaIndex3D( i, j, k, NODES_X )] );

	/* upload neighbouring buffers to GPU device */
	if ( ptr->ptrLeft not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_L, dev_velocity_u_s[cudaIndex3D( i-1, j, k, NODES_X )] );
		kernelCopyGrids __device_func__( velv_L, dev_velocity_v_s[cudaIndex3D( i-1, j, k, NODES_X )] );
		kernelCopyGrids __device_func__( velw_L, dev_velocity_w_s[cudaIndex3D( i-1, j, k, NODES_X )] );
		kernelCopyGrids __device_func__( dens_L,    dev_density_s[cudaIndex3D( i-1, j, k, NODES_X )] );
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_L );
		kernelZeroGrids __device_func__ ( velv_L );
		kernelZeroGrids __device_func__ ( velw_L );
		kernelZeroGrids __device_func__ ( dens_L );
	}

	if ( ptr->ptrRight not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_R, dev_velocity_u_s[cudaIndex3D( i+1, j, k, NODES_X )] );
		kernelCopyGrids __device_func__( velv_R, dev_velocity_v_s[cudaIndex3D( i+1, j, k, NODES_X )] );
		kernelCopyGrids __device_func__( velw_R, dev_velocity_w_s[cudaIndex3D( i+1, j, k, NODES_X )] );
		kernelCopyGrids __device_func__( dens_R,    dev_density_s[cudaIndex3D( i+1, j, k, NODES_X )] );
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_R );
		kernelZeroGrids __device_func__ ( velv_R );
		kernelZeroGrids __device_func__ ( velw_R );
		kernelZeroGrids __device_func__ ( dens_R );
	}

	if ( ptr->ptrUp not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_U, dev_velocity_u_s[cudaIndex3D( i, j+1, k, NODES_X )] );
		kernelCopyGrids __device_func__( velv_U, dev_velocity_v_s[cudaIndex3D( i, j+1, k, NODES_X )] );
		kernelCopyGrids __device_func__( velw_U, dev_velocity_w_s[cudaIndex3D( i, j+1, k, NODES_X )] );
		kernelCopyGrids __device_func__( dens_U,    dev_density_s[cudaIndex3D( i, j+1, k, NODES_X )] );
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_U );
		kernelZeroGrids __device_func__ ( velv_U );
		kernelZeroGrids __device_func__ ( velw_U );
		kernelZeroGrids __device_func__ ( dens_U );
	}

	if ( ptr->ptrDown not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_D, dev_velocity_u_s[cudaIndex3D( i, j-1, k, NODES_X )] );
		kernelCopyGrids __device_func__( velv_D, dev_velocity_v_s[cudaIndex3D( i, j-1, k, NODES_X )] );
		kernelCopyGrids __device_func__( velw_D, dev_velocity_w_s[cudaIndex3D( i, j-1, k, NODES_X )] );
		kernelCopyGrids __device_func__( dens_D,    dev_density_s[cudaIndex3D( i, j-1, k, NODES_X )] );
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_D );
		kernelZeroGrids __device_func__ ( velv_D );
		kernelZeroGrids __device_func__ ( velw_D );
		kernelZeroGrids __device_func__ ( dens_D );
	}

	if ( ptr->ptrFront not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_F, dev_velocity_u_s[cudaIndex3D( i, j, k+1, NODES_X )] );
		kernelCopyGrids __device_func__( velv_F, dev_velocity_v_s[cudaIndex3D( i, j, k+1, NODES_X )] );
		kernelCopyGrids __device_func__( velw_F, dev_velocity_w_s[cudaIndex3D( i, j, k+1, NODES_X )] );
		kernelCopyGrids __device_func__( dens_F,    dev_density_s[cudaIndex3D( i, j, k+1, NODES_X )] );
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_F );
		kernelZeroGrids __device_func__ ( velv_F );
		kernelZeroGrids __device_func__ ( velw_F );
		kernelZeroGrids __device_func__ ( dens_F );
	}

	if ( ptr->ptrBack not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_B, dev_velocity_u_s[cudaIndex3D( i, j, k-1, NODES_X )] );
		kernelCopyGrids __device_func__( velv_B, dev_velocity_v_s[cudaIndex3D( i, j, k-1, NODES_X )] );
		kernelCopyGrids __device_func__( velw_B, dev_velocity_w_s[cudaIndex3D( i, j, k-1, NODES_X )] );
		kernelCopyGrids __device_func__( dens_B,    dev_density_s[cudaIndex3D( i, j, k-1, NODES_X )] );
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_B );
		kernelZeroGrids __device_func__ ( velv_B );
		kernelZeroGrids __device_func__ ( velw_B );
		kernelZeroGrids __device_func__ ( dens_B );
	}
};

void FluidSimProc::SaveNode( int i, int j, int k )
{
	cudaDeviceDim3D();
	SimNode *ptr = gpu_node[cudaIndex3D( i, j, k, NODES_X )];

	/* draw data back */
	kernelCopyGrids __device_func__( dev_velocity_u_t[cudaIndex3D(i,j,k,NODES_X)], velu_C );
	kernelCopyGrids __device_func__( dev_velocity_v_t[cudaIndex3D(i,j,k,NODES_X)], velv_C );
	kernelCopyGrids __device_func__( dev_velocity_w_t[cudaIndex3D(i,j,k,NODES_X)], velw_C );
	kernelCopyGrids __device_func__(    dev_density_t[cudaIndex3D(i,j,k,NODES_X)], dens_C );

	/* draw volumetric data back */	
	kernelPickData __device_func__( dev_visual, dev_den, i * GRIDS_X, j * GRIDS_X, k * GRIDS_X );

	/* 将当前节点的标记设置为已更新 */
	ptr->updated = true;
};

void FluidSimProc::AddSource( void )
{
	if ( decrease_times eqt 0 )
	{
		cudaDeviceDim3D();
		kernelAddSource __device_func__ ( dev_den, dev_u, dev_v, dev_w, dev_obs );
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

	/* zero boundary buffers */
	kernelZeroGrids __device_func__ ( dev_obs );

	for ( int i = 0; i < host_obstacle.size(); i++ )
	{
		if ( cudaMemcpy( host_obstacle[i], dev_obs, m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			FreeResource();
			exit( 1 );
		}
	}	
	
	/* 将边界条件拷贝至内存 */
	kernelSetBoundary __device_func__( dev_obs );
	if ( cudaMemcpy( host_obstacle[cudaIndex3D(i,j,k,NODES_X)], dev_obs, m_node_size, cudaMemcpyDeviceToHost) not_eq cudaSuccess )
	{
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
		kernelZeroGrids  __device_func__ ( dev_buffers[i] );

	/* zero host buffer */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		kernelZeroGrids __device_func__ ( dev_density_s[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_u_s[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_v_s[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_w_s[i] );

		kernelZeroGrids __device_func__ ( dev_density_t[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_u_t[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_v_t[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_w_t[i] );
	}

	/* zero visual buffer */
	kernelZeroVolumetric __device_func__ ( dev_visual );
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );

	DownloadNodes();
};

void FluidSimProc::InteractNodes( int i, int j, int k )
{
	SimNode *ptr = gpu_node[cudaIndex3D(i,j,k,NODES_X)];
	int left, right, up, down, front, back;

	left = right = up = down = front = back = MACRO_FALSE;
	
	if ( ptr->ptrLeft  not_eq nullptr ) left  = ( (ptr->ptrLeft->updated) ? MACRO_TRUE : MACRO_FALSE );
	if ( ptr->ptrRight not_eq nullptr )	right = ( (ptr->ptrRight->updated)? MACRO_TRUE : MACRO_FALSE );
	if ( ptr->ptrUp    not_eq nullptr ) up    = ( (ptr->ptrUp->updated)   ? MACRO_TRUE : MACRO_FALSE );
	if ( ptr->ptrDown  not_eq nullptr )	down  = ( (ptr->ptrDown->updated) ? MACRO_TRUE : MACRO_FALSE );
	if ( ptr->ptrFront not_eq nullptr )	front = ( (ptr->ptrFront->updated)? MACRO_TRUE : MACRO_FALSE );
	if ( ptr->ptrBack  not_eq nullptr ) back  = ( (ptr->ptrBack->updated) ? MACRO_TRUE : MACRO_FALSE );

	cudaDeviceDim3D();
	kernelInteractNodes __device_func__
		( dens_C, dens_L, dens_R, dens_U, dens_D, dens_F, dens_B, left, right, up, down, front, back );
	kernelInteractNodes __device_func__
		( velu_C, velu_L, velu_R, velu_U, velu_D, velu_F, velu_B, left, right, up, down, front, back );
	kernelInteractNodes __device_func__
		( velv_C, velv_L, velv_R, velv_U, velv_D, velv_F, velv_B, left, right, up, down, front, back );
	kernelInteractNodes __device_func__
		( velw_C, velw_L, velw_R, velw_U, velw_D, velw_F, velw_B, left, right, up, down, front, back );
};

void FluidSimProc::Finally( FLUIDSPARAM *fluid )
{
	/* 更新节点数据 */
	cudaDeviceDim3D();	
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{		
		std::swap( dev_density_s[i], dev_density_t[i] );
		std::swap( dev_velocity_u_s[i], dev_velocity_u_t[i] );
		std::swap( dev_velocity_v_s[i], dev_velocity_v_t[i] );
		std::swap( dev_velocity_w_s[i], dev_velocity_w_t[i] );
		gpu_node[i]->updated = false;
	}

	/* 获取更新后的图形数据 */
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
	fluid->volume.ptrData = host_visual;

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
};