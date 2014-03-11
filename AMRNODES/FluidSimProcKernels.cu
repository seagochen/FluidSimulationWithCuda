/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 11, 2014
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

	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice( 0 ) != cudaSuccess ) { printf( "cudaSetDevice failed!\n" ); }

	/* allocate resources */
	if ( !AllocateResource() ) { FreeResource (); exit (1); }

	/* build order */
	CreateTopology();
	
	/* clear buffer */
	ZeroBuffers();

	/* set boundary */
	InitBoundary();

	/* upload host to device */
	HostToDevice();

	/* finally, print message */
	printf( "fluid simulation ready...\n" );
};

sstr FluidSimProc::GetTitleBar( void )
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

	m_cursor.x = m_cursor.y = m_cursor.z = 0;

	m_sz_title = "Excalibur OTL 3.00.00, AMR multi-nodes. ------------ FPS: %d ";
};

void FluidSimProc::CreateHostTopology( void )
{
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

				host_node[cudaIndex3D( i, j, k, HNODES_X )]->nodeIX.x = i;
				host_node[cudaIndex3D( i, j, k, HNODES_X )]->nodeIX.y = j;
				host_node[cudaIndex3D( i, j, k, HNODES_X )]->nodeIX.z = k;
			}
		}
	}
};

void FluidSimProc::CreateDeviceTopology( void )
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
};

void FluidSimProc::CreateTopology( void )
{
	CreateHostTopology();
	CreateDeviceTopology();
};

void FluidSimProc::PrintMSG( void )
{
	using namespace std;

	int totoal = 0;
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )	if ( host_node[i]->updated ) totoal++;

	cout 
		<< "**************** operation to confirm *******************" << endl
		<< "mouse wheel ------------ to rotate the observation matrix" << endl
		<< "keyboard: Q ------------ to quit the program" << endl
		<< "keyboard: Esc ---------- to quit the program" << endl
		<< "keyboard: C ------------ to clear the data of stage" << endl
		<< "keyboard: P ------------ to print message " << endl
		<< "keyboard: U ------------ to restore to previous" << endl
		<< "keyboard: D ------------ to keep current status in mem" << endl
		<< "**************** fluid simulation info ******************" << endl
		<< "no. of nodes for fluid simulation: " << gpu_node.size() << endl
		<< "no. of nodes for keeping the state of density : " << dev_density.size() << endl
		<< "no. of nodes for keeping the state of vel u : " << dev_velocity_u.size() << endl
		<< "no. of nodes for keeping the state of vel v : " << dev_velocity_v.size() << endl
		<< "no. of nodes for keeping the state of vel w : " << dev_velocity_w.size() << endl
		<< "no. of nodes for keeping pre state of density : " << host_density.size() << endl
		<< "no. of nodes for keeping pre state of vel u : " << host_velocity_u.size() << endl
		<< "no. of nodes for keeping pre state of vel v : " << host_velocity_v.size() << endl
		<< "no. of nodes for keeping pre state of vel w : " << host_velocity_w.size() << endl
		<< "grid dim per node : 64 x 64 x 64" << endl
		<< "updated nodes :" << totoal << endl;
};

bool FluidSimProc::CreateHostNodes( void )
{
	/* host nodes' topology */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		host_node.push_back( node );
	}

	/* vector for host nodes */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
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

	/* copy of host nodes */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		dev_density.push_back( ptrDens );
		dev_velocity_u.push_back( ptrU );
		dev_velocity_v.push_back( ptrV );
		dev_velocity_w.push_back( ptrW );
		dev_obstacle.push_back( ptrObs );
	}

	return true;
};

bool FluidSimProc::CreateDeviceNodes( void )
{
	/* device nodes' topology */
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		gpu_node.push_back( node );
	}

	/* vector for device nodes */
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		node_density.push_back( ptrDens );
		node_velocity_u.push_back( ptrU );
		node_velocity_v.push_back( ptrV );
		node_velocity_w.push_back( ptrW );
		node_obstacle.push_back( ptrObs );
	}
	
	/* global node */
	double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;
	node_density.push_back( ptrDens );
	node_velocity_u.push_back( ptrU );
	node_velocity_v.push_back( ptrV );
	node_velocity_w.push_back( ptrW );
	node_obstacle.push_back( ptrObs );

	return true;
};

bool FluidSimProc::CreateTempBuffers( void )
{
	/* tempororay buffers reserved */
	if ( helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(double), 1, &dev_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(int), 1, &dev_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_X*sizeof(double), 1, &host_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_X*sizeof(int), 1, &host_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;

	/* allocate memory on GPU devices */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		dev_buffers.push_back(ptr);
	}

	return true;
};

bool FluidSimProc::CreateVolumeBuffers( void )
{
	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )  return false;

	return true;
};

bool FluidSimProc::AllocateResource ( void )
{
	/* allocating resource */
	if ( !CreateHostNodes() ) 
	{
		printf( "create host nodes failed!\n" );
		return false;
	}

	if ( !CreateDeviceNodes() )
	{
		printf( "create device nodes failed!\n" );
		return false;
	}

	if ( !CreateTempBuffers() )
	{
		printf( "create temporary buffers failed!\n" );
		return false;
	}

	if ( !CreateVolumeBuffers() )
	{
		printf( "create volumetric buffers failed!\n" );
		return false;
	}
	
	/* check the CUDA device if something occured */
	if ( helper.GetCUDALastError( "memory allocation failed, check the code",
		__FILE__, __LINE__ ) ) return false;
	
	/* finally */
	return true;
};

void FluidSimProc::FreeHostNodes( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		helper.FreeHostBuffers( 1, &host_density[i] );
		helper.FreeHostBuffers( 1, &host_velocity_u[i] );
		helper.FreeHostBuffers( 1, &host_velocity_v[i] );
		helper.FreeHostBuffers( 1, &host_velocity_w[i] );
		helper.FreeHostBuffers( 1, &host_obstacle[i] );
	}

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		helper.FreeDeviceBuffers( 1, &dev_density[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_u[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_v[i] );
		helper.FreeDeviceBuffers( 1, &dev_velocity_w[i] );
		helper.FreeDeviceBuffers( 1, &dev_obstacle[i] );
	}
};

void FluidSimProc::FreeDeviceNodes( void )
{
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X + 1; i++ )
	{
		helper.FreeDeviceBuffers( 1, &node_obstacle[i] );
		helper.FreeDeviceBuffers( 1, &node_density[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_u[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_v[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_w[i] );	
	}
};

void FluidSimProc::FreeTempBuffers( void )
{
	for ( int i = 0; i < dev_buffers_num; i++ ) 
		helper.FreeDeviceBuffers( 1, &dev_buffers[i] );

	helper.FreeDeviceBuffers( 2, &dev_dtpbuf, &dev_ntpbuf );
	helper.FreeHostBuffers( 2, &host_dtpbuf, &host_ntpbuf );
};

void FluidSimProc::FreeVolumeBuffers( void )
{
	helper.FreeDeviceBuffers( 1, &dev_visual );
	helper.FreeHostBuffers( 1, &host_visual );
};

void FluidSimProc::FreeResource ( void )
{
	FreeHostNodes();
	FreeDeviceNodes();
	FreeTempBuffers();
	FreeVolumeBuffers();
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

void FluidSimProc::ClearBullet( void )
{
	cudaDeviceDim3D();
	
	kernelZeroGrids __device_func__ ( velu_L );
	kernelZeroGrids __device_func__ ( velv_L );
	kernelZeroGrids __device_func__ ( velw_L );
	kernelZeroGrids __device_func__ ( dens_L );

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	kernelZeroGrids __device_func__ ( velu_R );
	kernelZeroGrids __device_func__ ( velv_R );
	kernelZeroGrids __device_func__ ( velw_R );
	kernelZeroGrids __device_func__ ( dens_R );

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	kernelZeroGrids __device_func__ ( velu_U );
	kernelZeroGrids __device_func__ ( velv_U );
	kernelZeroGrids __device_func__ ( velw_U );
	kernelZeroGrids __device_func__ ( dens_U );

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	kernelZeroGrids __device_func__ ( velu_D );
	kernelZeroGrids __device_func__ ( velv_D );
	kernelZeroGrids __device_func__ ( velw_D );
	kernelZeroGrids __device_func__ ( dens_D );

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	kernelZeroGrids __device_func__ ( velu_F );
	kernelZeroGrids __device_func__ ( velv_F );
	kernelZeroGrids __device_func__ ( velw_F );
	kernelZeroGrids __device_func__ ( dens_F );

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	kernelZeroGrids __device_func__ ( velu_B );
	kernelZeroGrids __device_func__ ( velv_B );
	kernelZeroGrids __device_func__ ( velw_B );
	kernelZeroGrids __device_func__ ( dens_B );

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::LoadBullet( int i, int j, int k )
{
	ClearBullet();

	/* upload center node to GPU device */
	cudaDeviceDim3D();
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

	SimNode *ptr = gpu_node[cudaIndex3D( i, j, k, GNODES_X )];

	/* upload neighbouring buffers to GPU device */
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

void FluidSimProc::ExitBullet( int i, int j, int k )
{
	/* draw data back */
	cudaDeviceDim3D();
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

void FluidSimProc::AddSource( void )
{
	if ( decrease_times eqt 0 )
	{
		cudaDeviceDim3D();
		kernelAddSource __device_func__ ( dev_den, dev_u, dev_v, dev_w, dev_obs );

		if ( helper.GetCUDALastError( "device kernel: kernelPickData failed", __FILE__, __LINE__ ) )
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

//TODO 采用AMR方法后，该函数将清空，为以后开发做准备
void FluidSimProc::InitBoundary( void )
{
	cudaDeviceDim3D();

	/* zero boundary buffers */
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
	
	/* set boundary condition */
	kernelSetBoundary __device_func__( dev_obs );
#if HNODES_X == 3
	if ( cudaMemcpy( host_obstacle[cudaIndex3D(1,0,1,HNODES_X)], dev_obs, m_node_size, cudaMemcpyDeviceToHost) not_eq cudaSuccess )
	{
		helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
#else
	if ( cudaMemcpy( host_obstacle[cudaIndex3D(0,0,0,HNODES_X)], dev_obs, m_node_size, cudaMemcpyDeviceToHost) not_eq cudaSuccess )
	{
		helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
#endif
};

void FluidSimProc::VelocitySolver( cdouble delta )
{
	// diffuse the velocity field (per axis):
	Diffusion( dev_u0, dev_u, VISOCITY, dev_obs, MACRO_VELOCITY_U );
	Diffusion( dev_v0, dev_v, VISOCITY, dev_obs, MACRO_VELOCITY_V );
	Diffusion( dev_w0, dev_w, VISOCITY, dev_obs, MACRO_VELOCITY_W );
	
	if ( helper.GetCUDALastError( "host function failed: hostDiffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection();

	if ( helper.GetCUDALastError( "host function failed: hostProject", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	Advection( dev_u0, dev_u, delta, dev_obs, MACRO_VELOCITY_U, dev_u, dev_v, dev_w );
	Advection( dev_v0, dev_v, delta, dev_obs, MACRO_VELOCITY_V, dev_u, dev_v, dev_w );
	Advection( dev_w0, dev_w, delta, dev_obs, MACRO_VELOCITY_W, dev_u, dev_v, dev_w );

	if ( helper.GetCUDALastError( "host function failed: hostAdvection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection();
};

void FluidSimProc::DensitySolver( cdouble delta )
{
	Diffusion( dev_den0, dev_den, DIFFUSION, dev_obs, MACRO_DENSITY );
	std::swap( dev_den0, dev_den );
	Advection ( dev_den, dev_den0, delta, dev_obs, MACRO_DENSITY, dev_u, dev_v, dev_w );

	if ( helper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::ZeroDeviceBuffers( void )
{
	cudaDeviceDim3D();

	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X + 1; i++ )
	{
		kernelZeroGrids __device_func__ ( node_density[i] );
		kernelZeroGrids __device_func__ ( node_velocity_u[i] );
		kernelZeroGrids __device_func__ ( node_velocity_v[i] );
		kernelZeroGrids __device_func__ ( node_velocity_w[i] );
	}
};

void FluidSimProc::ZeroHostBuffers( void )
{
	cudaDeviceDim3D();

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		kernelZeroGrids __device_func__ ( dev_density[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_u[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_v[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_w[i] );
	}
};

void FluidSimProc::ZeroTempBuffers( void )
{
	cudaDeviceDim1D();

	kernelZeroTemporaryBuffers __device_func__ ( dev_ntpbuf );
	kernelZeroTemporaryBuffers __device_func__ ( dev_dtpbuf );
	
	/* zero GPU buffer */
	for ( int i = 0; i < dev_buffers_num; i++ )
		kernelZeroGrids  __device_func__ ( dev_buffers[i] );
};

//TODO 由于新方法中，volumetric data数据维数不再是（64）的倍数，而是（62）的倍数。
//因此对于volumetric数据的清零，将另外写一个Kernel函数
void FluidSimProc::ZeroVolumeBuffers( void )
{
	/* zero visual buffer */
	kernelZeroVolumetric __device_func__ ( dev_visual );
};

void FluidSimProc::ZeroBuffers( void )
{
	ZeroTempBuffers();
	ZeroDeviceBuffers();
	ZeroHostBuffers();
	ZeroVolumeBuffers();
	
	if ( helper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		cudaMemcpy( host_density[i], dev_density[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_u[i], dev_velocity_u[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_velocity_v[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_velocity_w[i], m_node_size, cudaMemcpyDeviceToHost );
	}
	cudaMemcpy( host_ntpbuf, dev_ntpbuf, sizeof(int)*TPBUFFER_X, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_dtpbuf, dev_dtpbuf, sizeof(double)*TPBUFFER_X, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );

	if ( helper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

//TODO 该方法原为各节点之间交换数据时使用，但使用AMR方法后，将采用全新的数据交换方法，所有当其他函数
//大体上完成后，该方法将被删除
void FluidSimProc::Interaction( int i, int j, int k )
{
	SimNode *ptr = gpu_node[cudaIndex3D(i,j,k,GNODES_X)];
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

void FluidSimProc::RefreshFPS( FLUIDSPARAM *fluid )
{
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

void FluidSimProc::RefreshHostNodes( FLUIDSPARAM *fluid )
{
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
		host_node[i]->updated = false;
};

void FluidSimProc::RefreshStatus( FLUIDSPARAM *fluid )
{
	RefreshFPS( fluid );
	RefreshVolume( fluid );
	RefreshHostNodes( fluid );
};

//TODO 全局数据生成完毕后，要依据host节点的数据，生成最后的3D volume data
void FluidSimProc::RefreshVolume( FLUIDSPARAM *fluid )
{
	/* waiting for all kernels end */
	if ( cudaThreadSynchronize() not_eq cudaSuccess )
	{
		printf( "cudaThreadSynchronize failed\n" );
		FreeResource();
		exit( 1 );
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

void FluidSimProc::SetCursor( int i, int j, int k )
{
	m_cursor.x = i;
	m_cursor.y = j;
	m_cursor.z = k;
};

void FluidSimProc::ReadBuffers( void )
{
	int ni, nj, nk; ni = nj = nk = 0;

	cudaDeviceDim3D();

	for ( int k = 0; k < GNODES_X; k++ ) for ( int j = 0; j < GNODES_X; j++ ) for ( int i = 0; i < GNODES_X; i++ )
	{
		ni = m_cursor.x + i;
		nj = m_cursor.y + j;
		nk = m_cursor.z + k;

		/* load node status */
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->nodeIX.x = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->nodeIX.x;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->nodeIX.x = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->nodeIX.x;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->nodeIX.x = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->nodeIX.x;
		gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated  = host_node[cudaIndex3D(ni,nj,nk,HNODES_X)]->updated;

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

		/* updated the data */

		kernelCopyGrids __device_func__ ( dev_density[cudaIndex3D(ni,nj,nk,HNODES_X)], node_density[cudaIndex3D(i,j,k,GNODES_X)] );
		kernelCopyGrids __device_func__ ( dev_velocity_u[cudaIndex3D(ni,nj,nk,HNODES_X)], node_velocity_u[cudaIndex3D(i,j,k,GNODES_X)] );
		kernelCopyGrids __device_func__ ( dev_velocity_v[cudaIndex3D(ni,nj,nk,HNODES_X)], node_velocity_v[cudaIndex3D(i,j,k,GNODES_X)] );
		kernelCopyGrids __device_func__ ( dev_velocity_w[cudaIndex3D(ni,nj,nk,HNODES_X)], node_velocity_w[cudaIndex3D(i,j,k,GNODES_X)] );
	}
};

void FluidSimProc::Jacobi
	( double *out, cdouble *in, cdouble *obstacle, cint field, cdouble diffusion, cdouble divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi<<<gridDim,blockDim>>>( out, in, diffusion, divisor );
	}
	kernelObstacle<<<gridDim,blockDim>>>( out, obstacle, field );
};

void FluidSimProc::Advection
	( double *out, cdouble *in, cdouble delta, cdouble *obstacle, cint field, cdouble *u, cdouble *v, cdouble *w )
{
	cudaDeviceDim3D();
	kernelGridAdvection<<<gridDim,blockDim>>>( out, in, delta, u, v, w );
	kernelObstacle<<<gridDim,blockDim>>>( out, obstacle, field );
};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diffusion, cdouble *obstacle, cint field )
{
	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	Jacobi ( out, in, obstacle, field, rate, 1+6*rate );
};

void FluidSimProc::Projection( void )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient<<<gridDim,blockDim>>>( dev_div, dev_p, dev_u, dev_v, dev_w );
	kernelObstacle<<<gridDim,blockDim>>>( dev_div, dev_obs, MACRO_SIMPLE );
	kernelObstacle<<<gridDim,blockDim>>>( dev_p, dev_obs, MACRO_SIMPLE );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(dev_p, dev_div, dev_obs, MACRO_SIMPLE, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract<<<gridDim,blockDim>>>( dev_u, dev_v, dev_w, dev_p );
	kernelObstacle<<<gridDim,blockDim>>>( dev_u, dev_obs, MACRO_VELOCITY_U );
	kernelObstacle<<<gridDim,blockDim>>>( dev_v, dev_obs, MACRO_VELOCITY_V );
	kernelObstacle<<<gridDim,blockDim>>>( dev_w, dev_obs, MACRO_VELOCITY_W );
};

//TODO 当SolveLeafGrids函数完成后，该方法将删除掉
void FluidSimProc::SolveNavierStokers( void )
{
	/* updating */
	for ( int i = 0; i < GNODES_X; i++ )
	{
		for ( int j = 0; j < GNODES_X; j++ )
		{
			for ( int k = 0; k < GNODES_X; k++ )
			{
				if ( !gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated )
				{
					LoadBullet(i,j,k);					
					Interaction(i,j,k);									
					AddSource();
					VelocitySolver( DELTATIME );
					DensitySolver( DELTATIME );
					ExitBullet(i,j,k);
					gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated = true;
				}
			}
		}
	}
};

//TODO 对全局节点的数据采集
void FluidSimProc::InterRootGrids( void )
{
};

//TODO 对子节点的数据采集
void FluidSimProc::InterLeafGrids( void )
{
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	SolveRootGrids();

	SolveLeafGrids();
	
	RefreshStatus( fluid );
};

//TODO 当全局节点完成一次计算后，需要与邻近的节点交换ghost cell的数据，之后进行一次delta_time / 2的计算
// 当所有子节点计算完毕后，需要进行一次数据采集，并将更新的数据写回root节点，为下一次计算做准备
void FluidSimProc::SolveLeafGrids( void )
{
	for ( int k = 0; k < CURSOR_X; k++ )
	{
		for ( int j = 0; j < CURSOR_X; j++ )
		{
			for ( int i = 0; i < CURSOR_X; i++ )
			{
				SetCursor( i, j, k );				
				ReadBuffers();
				SolveNavierStokers();
				WriteBuffers();
			}
		}
	}
};

//TODO 对全局节点先进性一次delta time的计算，然后将全局的数据依次采集到各个独立的节点中
void FluidSimProc::SolveRootGrids( void )
{
	cudaDeviceDim3D();

	kernelCopyGrids __device_func__ ( dev_den, node_density[GLOBAL] );
	kernelCopyGrids __device_func__ ( dev_u, node_velocity_u[GLOBAL] );
	kernelCopyGrids __device_func__ ( dev_v, node_velocity_v[GLOBAL] );
	kernelCopyGrids __device_func__ ( dev_w, node_velocity_w[GLOBAL] );

	AddSource();
	VelocitySolver( DELTATIME );
	DensitySolver( DELTATIME );
};