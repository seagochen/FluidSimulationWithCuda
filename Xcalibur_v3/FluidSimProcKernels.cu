/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 19, 2014
* <File Name>     FluidSimProcKernels.cu
*/

#include <time.h>
#include <iostream>
#include <utility>
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "FrameworkDynamic.h"
#include "Kernels.h"

using namespace sge;

#define Index(i,j,k,ts) (k)*(ts)*(ts)+(j)*(ts)+i
#define __device_func__ <<<gridDim, blockDim>>>


FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	/* initialize FPS */
	InitParams( fluid );

	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
	{
		printf( "cudaSetDevice failed!\n" );
		exit(1);
	}

	/* allocate resources */
	if ( !AllocateResource() ) 
	{
		FreeResource();
		exit(1);
	}

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

	m_node_size = GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double);
	m_volm_size = VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR);

	increase_times = decrease_times = 0;

	ptr = nullptr;
	m_cursor.x = m_cursor.y = m_cursor.z = 0;

	m_sz_title = "Excalibur OTL 2.10.00, large-scale. ------------ FPS: %d ";
};

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
					gpu_node[Index( i, j, k, GNODES_X )]->ptrLeft  = gpu_node[Index( i-1, j, k, GNODES_X )];
				/* right */
				if ( i <= GNODES_X - 2 )
					gpu_node[Index( i, j, k, GNODES_X )]->ptrRight = gpu_node[Index( i+1, j, k, GNODES_X )];
				/* down */
				if ( j >= 1 )
					gpu_node[Index( i, j, k, GNODES_X )]->ptrDown  = gpu_node[Index( i, j-1, k, GNODES_X )];
				/* up */
				if ( j <= GNODES_X - 2 )
					gpu_node[Index( i, j, k, GNODES_X )]->ptrUp    = gpu_node[Index( i, j+1, k, GNODES_X )];
				/* back */
				if ( k >= 1 )
					gpu_node[Index( i, j, k, GNODES_X )]->ptrBack  = gpu_node[Index( i, j, k-1, GNODES_X )];
				/* front */
				if ( k <= GNODES_X - 2 )
					gpu_node[Index( i, j, k, GNODES_X )]->ptrFront = gpu_node[Index( i, j, k+1, GNODES_X )];
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
					host_node[Index( i, j, k, HNODES_X )]->ptrLeft  = host_node[Index( i-1, j, k, HNODES_X )];
				/* right */
				if ( i <= HNODES_X - 2 )
					host_node[Index( i, j, k, HNODES_X )]->ptrRight = host_node[Index( i+1, j, k, HNODES_X )];
				/* down */
				if ( j >= 1 )
					host_node[Index( i, j, k, HNODES_X )]->ptrDown  = host_node[Index( i, j-1, k, HNODES_X )];
				/* up */
				if ( j <= HNODES_X - 2 )
					host_node[Index( i, j, k, HNODES_X )]->ptrUp    = host_node[Index( i, j+1, k, HNODES_X )];
				/* back */
				if ( k >= 1 )
					host_node[Index( i, j, k, HNODES_X )]->ptrBack  = host_node[Index( i, j, k-1, HNODES_X )];
				/* front */
				if ( k <= HNODES_X - 2 )
					host_node[Index( i, j, k, HNODES_X )]->ptrFront = host_node[Index( i, j, k+1, HNODES_X )];

				host_node[Index( i, j, k, HNODES_X )]->x = i;
				host_node[Index( i, j, k, HNODES_X )]->y = j;
				host_node[Index( i, j, k, HNODES_X )]->z = k;
			}
		}
	}
};

void FluidSimProc::PrintMSG( void )
{
	using namespace std;

	int totoal = 0;
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )	if ( host_node[i]->updated ) totoal++;

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

bool FluidSimProc::allocHostRes( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
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

	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
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

bool FluidSimProc::allocDeviceRes( void )
{
	for ( int i = 0; i < GNODES_X * GNODES_Y * GNODES_Z; i++ )
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

	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		dev_buffers.push_back(ptr);
	}

	if ( helper.CreateDeviceBuffers( m_node_size, 1, &gd_density ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &gd_obstacle ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &gd_velocity_u ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &gd_velocity_v ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( m_node_size, 1, &gd_velocity_w ) not_eq SG_RUNTIME_OK ) return false;
	
	return true;
};

bool FluidSimProc::allocShareBuffers( void )
{
	if ( helper.CreateDeviceBuffers( TPBUFFER_S*sizeof(double), 1, &dev_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( TPBUFFER_S*sizeof(int), 1, &dev_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_S*sizeof(double), 1, &host_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_S*sizeof(int), 1, &host_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;

	return true;
};

bool FluidSimProc::allocVisualBuffers( void )
{
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )  return false;

	return true;
};

void FluidSimProc::allocTopologyNodes( void )
{
	for ( int i = 0; i < GNODES_X * GNODES_Y * GNODES_Z; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		node->active   = false;
		gpu_node.push_back( node );
	}

	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		node->active   = false;
		host_node.push_back( node );
	}
};

bool FluidSimProc::AllocateResource( void )
{
	if ( !allocHostRes() )
	{
		printf( "allocate resource for host nodes failed!\n" );
		return false;
	}

	if ( !allocDeviceRes() ) 
	{
		printf( "allocate resource for device nodes failed\n" );
		return false;
	}

	if ( !allocShareBuffers() )
	{
		printf( "allocate resource for sharing memories failed!\n" );
		return false;
	}

	if ( !allocVisualBuffers() )
	{
		printf( "allocate resource for visual buffers failed!\n" );
		return false;
	}
	
	allocTopologyNodes();

	if ( helper.GetCUDALastError( "memory allocation failed, check the code", __FILE__, __LINE__ ) )
		return false;
	
	return true;
}

void FluidSimProc::freeHostRes( void )
{
	/* free node resource */
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
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
	for ( int i = 0; i < GNODES_X * GNODES_Y * GNODES_Z; i++ )
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
//	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		kernelZeroBuffers __device_func__ ( dev_buffers[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};

void FluidSimProc::zeroGlobalNode( void )
{
//	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	 kernelZeroBuffers __device_func__ ( gd_density, GRIDS_X, GRIDS_Y, GRIDS_Z );
	 kernelZeroBuffers __device_func__ ( gd_velocity_u, GRIDS_X, GRIDS_Y, GRIDS_Z );
	 kernelZeroBuffers __device_func__ ( gd_velocity_v, GRIDS_X, GRIDS_Y, GRIDS_Z );
	 kernelZeroBuffers __device_func__ ( gd_velocity_w, GRIDS_X, GRIDS_Y, GRIDS_Z );
};

void FluidSimProc::zeroDeivceRes( void )
{
//	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	for ( int i = 0; i < GNODES_X * GNODES_Y * GNODES_Z; i++ )
	{	
		kernelZeroBuffers __device_func__ ( node_density[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( node_velocity_u[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( node_velocity_v[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( node_velocity_w[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	
	}
};

void FluidSimProc::zeroHostRes( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_Y * HNODES_Z; i++ )
	{
		kernelZeroBuffers __device_func__ ( dev_density[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( dev_velocity_u[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( dev_velocity_v[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( dev_velocity_w[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};

void FluidSimProc::zeroVisualBuffers( void )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, VOLUME_X, VOLUME_Y, VOLUME_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_X, VOLUME_X, VOLUME_Y, VOLUME_Z );

	kernelZeroBuffers __device_func__ ( dev_visual, VOLUME_X, VOLUME_Y, VOLUME_Z );
};

void FluidSimProc::zeroShareBuffers( void )
{
//	helper.DeviceDim1D( &blockDim, &gridDim, THREADS_S, TPBUFFER_S );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TPBUFFER_S );

	kernelZeroBuffers __device_func__ ( dev_ntpbuf, TPBUFFER_S );
	kernelZeroBuffers __device_func__ ( dev_dtpbuf, TPBUFFER_S );
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
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int i = 11; i < dev_buffers_num; i++ )
	{
		kernelZeroBuffers __device_func__ ( dev_buffers[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickNodeToBullet( int i, int j, int k )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	/* upload center node to GPU device */
	kernelCopyBuffers __device_func__ ( dev_u, node_velocity_u[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_v, node_velocity_v[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_w, node_velocity_w[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_den,  node_density[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( dev_obs, node_obstacle[Index( i, j, k, GNODES_X )], GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickNeighborsToBullet( int i, int j, int k )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

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
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelCopyBuffers __device_func__ ( node_velocity_u[Index(i,j,k,GNODES_X)], velu_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( node_velocity_v[Index(i,j,k,GNODES_X)], velv_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( node_velocity_w[Index(i,j,k,GNODES_X)], velw_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelCopyBuffers __device_func__ ( node_density[Index(i,j,k,GNODES_X)],    dens_C, GRIDS_X, GRIDS_Y, GRIDS_Z );
	
	if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::pickImgFromNode( int i, int j, int k )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

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
		helper.GetCUDALastError( "host function: cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::InitBoundary( void )
{
//	 helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	/* zero boundary buffers */
	 kernelZeroBuffers __device_func__ ( dev_obs, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int i = 0; i < host_obstacle.size(); i++ )
	{
		if ( cudaMemcpy( host_obstacle[i], dev_obs, m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}
	}

	//kernelCopyGrids __device_func__ ( gd_obstacle, dev_obs );
	kernelCopyBuffers __device_func__ ( gd_obstacle, dev_obs, GRIDS_X, GRIDS_Y, GRIDS_Z );

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
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

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

		kernelInterRootGrids __device_func__ ( dev_density[Index(i,j,k,HNODES_X)], dev_den, i, j, k, rate );
		kernelInterRootGrids __device_func__ ( dev_velocity_u[Index(i,j,k,HNODES_X)], dev_u, i, j, k, rate );
		kernelInterRootGrids __device_func__ ( dev_velocity_v[Index(i,j,k,HNODES_X)], dev_v, i, j, k, rate );
		kernelInterRootGrids __device_func__ ( dev_velocity_w[Index(i,j,k,HNODES_X)], dev_w, i, j, k, rate );
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
		kernelInterLeafGrids __device_func__ ( gd_density, dev_density[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_u, dev_velocity_u[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_v, dev_velocity_v[i], ptr->x, ptr->y, ptr->z, rate );
		kernelInterLeafGrids __device_func__ ( gd_velocity_w, dev_velocity_w[i], ptr->x, ptr->y, ptr->z, rate );
	}

	for ( int k = 0; k < HNODES_X; k++ ) for ( int j = 0; j < HNODES_X; j++ ) for ( int i = 0; i < HNODES_X; i++ )
	{
		kernelPickData __device_func__ ( dev_visual, dev_density[Index(i,j,k,HNODES_X)], i, j, k, GRIDS_X );
	}
};

void FluidSimProc::SolveLeafNode( void )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

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

					kernelHandleHalo __device_func__ ( dev_den, dens_L, dens_R, dens_U, dens_D, dens_F, dens_B );
					kernelHandleHalo __device_func__ ( dev_u,   velu_L, velu_R, velu_U, velu_D, velu_F, velu_B );
					kernelHandleHalo __device_func__ ( dev_v,   velv_L, velv_R, velv_U, velv_D, velv_F, velv_B );
					kernelHandleHalo __device_func__ ( dev_w,   velw_L, velw_R, velw_U, velw_D, velw_F, velw_B );

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

void FluidSimProc::AddSource( void )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( decrease_times eqt 0 )
	{
		kernelAddSource __device_func__ ( dev_den, dev_u, dev_v, dev_w );

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
	
	if ( helper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );

	if ( helper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	Advection( dev_u0, dev_u, timestep, dev_u, dev_v, dev_w );
	Advection( dev_v0, dev_v, timestep, dev_u, dev_v, dev_w );
	Advection( dev_w0, dev_w, timestep, dev_u, dev_v, dev_w );

	if ( helper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
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

	if ( helper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	for ( int k=0; k<20; k++)
	{
		kernelJacobi<<<gridDim,blockDim>>>( out, in, diff, divisor);
	}
};

void FluidSimProc::Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	kernelGridAdvection<<<gridDim,blockDim>>>( out, in, timestep, u, v, w );
};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_Y * GRIDS_Z;
	Jacobi ( out, in, rate, 1+6*rate );
};

void FluidSimProc::Projection( double *u, double *v, double *w, double *div, double *p )
{
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


	// the velocity gradient
	kernelGradient<<<gridDim,blockDim>>>( div, p, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract<<<gridDim,blockDim>>>( u, v, w, p );
};

void FluidSimProc::ReadBuffers( void )
{
	int ni, nj, nk;
	ni = nj = nk = 0;
//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


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

//	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_S, TILE_X, GRIDS_X, GRIDS_Y, GRIDS_Z );
	helper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );


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