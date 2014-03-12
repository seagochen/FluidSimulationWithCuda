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


void FluidSimProc::hostJacobi( double *grid_out, cdouble *grid_in, cdouble *obstacle, cint field, cdouble diffusion, cdouble divisor )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	for ( int k=0; k<20; k++)
	{
		kernelJacobi<<<gridDim,blockDim>>>(grid_out, grid_in, diffusion, divisor);
	}
	kernelObstacle<<<gridDim,blockDim>>>( grid_out, obstacle, field );
};

void FluidSimProc::hostAdvection( double *grid_out, cdouble *grid_in, cdouble *obstacle, cint field, cdouble *u_in, cdouble *v_in, cdouble *w_in )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	kernelGridAdvection<<<gridDim,blockDim>>>( grid_out, grid_in, DELTATIME, u_in, v_in, w_in );
	kernelObstacle<<<gridDim,blockDim>>>( grid_out, obstacle, field );
};

void FluidSimProc::hostDiffusion( double *grid_out, cdouble *grid_in, cdouble diffusion, cdouble *obstacle, cint field )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, obstacle, field, rate, 1+6*rate );
};

void FluidSimProc::hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p, cdouble *obs )
{
	helper.DeviceDim3D( &blockDim, &gridDim, THREADS_X, TILE_X, GRIDS_X, GRIDS_X, GRIDS_X );

	// the velocity gradient
	kernelGradient<<<gridDim,blockDim>>>( div, p, vel_u, vel_v, vel_w );
	kernelObstacle<<<gridDim,blockDim>>>( div, obs, MACRO_SIMPLE );
	kernelObstacle<<<gridDim,blockDim>>>( p, obs, MACRO_SIMPLE );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, obs, MACRO_SIMPLE, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract<<<gridDim,blockDim>>>( vel_u, vel_v, vel_w, p );
	kernelObstacle<<<gridDim,blockDim>>>( vel_u, obs, MACRO_VELOCITY_U );
	kernelObstacle<<<gridDim,blockDim>>>( vel_v, obs, MACRO_VELOCITY_V );
	kernelObstacle<<<gridDim,blockDim>>>( vel_w, obs, MACRO_VELOCITY_W );
};

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
	InitBoundary();

	/* upload host to device */
	IO_UploadBuffers();

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

				host_node[cudaIndex3D( i, j, k, HNODES_X )]->nodeIX.x = i;
				host_node[cudaIndex3D( i, j, k, HNODES_X )]->nodeIX.y = j;
				host_node[cudaIndex3D( i, j, k, HNODES_X )]->nodeIX.z = k;
			}
		}
	}
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

bool FluidSimProc::AllocateResource ( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess ) return false; 

	/* tempororay buffers reserved */
	if ( helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(double), 1, &dev_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(int), 1, &dev_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_X*sizeof(double), 1, &host_dtpbuf ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( TPBUFFER_X*sizeof(int), 1, &host_ntpbuf ) not_eq SG_RUNTIME_OK ) return false;

	/* vector of node topology */
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		gpu_node.push_back( node );
	}

	/* vector of node topology */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		host_node.push_back( node );
	}

	/* vector of fluid simulation buffers */
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

	/* create GPU node for fluid simulation */
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

	/* allocate memory on GPU devices */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		dev_buffers.push_back(ptr);
	}

	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )  return false;

	/* check the CUDA device if something occured */
	if ( helper.GetCUDALastError( "memory allocation failed, check the code", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	/* finally */
	return true;
}  

void FluidSimProc::FreeResource ( void )
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
	}

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		/* release device resource */
		helper.FreeDeviceBuffers( 1, &host_density[i] );
		helper.FreeDeviceBuffers( 1, &host_velocity_u[i] );
		helper.FreeDeviceBuffers( 1, &host_velocity_v[i] );
		helper.FreeDeviceBuffers( 1, &host_velocity_w[i] );
		helper.FreeDeviceBuffers( 1, &host_obstacle[i] );
	}

	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{
		/* release device resource */
		helper.FreeDeviceBuffers( 1, &node_obstacle[i] );
		helper.FreeDeviceBuffers( 1, &node_density[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_u[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_v[i] );
		helper.FreeDeviceBuffers( 1, &node_velocity_w[i] );	
	}

	/* free device resource */
	for ( int i = 0; i < dev_buffers_num; i++ ) 
		helper.FreeDeviceBuffers( 1, &dev_buffers[i] );
}

void FluidSimProc::IO_UploadBuffers( void )
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

void FluidSimProc::IO_DownloadBuffers( void )
{
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		cudaMemcpy( host_density[i], dev_density[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_u[i], dev_velocity_u[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_velocity_v[i], m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_velocity_w[i], m_node_size, cudaMemcpyDeviceToHost );
	}
};

void FluidSimProc::LoadNode( int i, int j, int k )
{
	helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );
	SimNode *ptr = gpu_node[cudaIndex3D( i, j, k, GNODES_X )];

	/* upload center node to GPU device */
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

	/* upload neighbouring buffers to GPU device */
	if ( ptr->ptrLeft not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_L, node_velocity_u[cudaIndex3D( i-1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_L, node_velocity_v[cudaIndex3D( i-1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_L, node_velocity_w[cudaIndex3D( i-1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_L,    node_density[cudaIndex3D( i-1, j, k, GNODES_X )] );

		if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_L );
		kernelZeroGrids __device_func__ ( velv_L );
		kernelZeroGrids __device_func__ ( velw_L );
		kernelZeroGrids __device_func__ ( dens_L );

		if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrRight not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_R, node_velocity_u[cudaIndex3D( i+1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_R, node_velocity_v[cudaIndex3D( i+1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_R, node_velocity_w[cudaIndex3D( i+1, j, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_R,    node_density[cudaIndex3D( i+1, j, k, GNODES_X )] );

		if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_R );
		kernelZeroGrids __device_func__ ( velv_R );
		kernelZeroGrids __device_func__ ( velw_R );
		kernelZeroGrids __device_func__ ( dens_R );

		if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrUp not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_U, node_velocity_u[cudaIndex3D( i, j+1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_U, node_velocity_v[cudaIndex3D( i, j+1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_U, node_velocity_w[cudaIndex3D( i, j+1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_U,    node_density[cudaIndex3D( i, j+1, k, GNODES_X )] );

		if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_U );
		kernelZeroGrids __device_func__ ( velv_U );
		kernelZeroGrids __device_func__ ( velw_U );
		kernelZeroGrids __device_func__ ( dens_U );

		if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrDown not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_D, node_velocity_u[cudaIndex3D( i, j-1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_D, node_velocity_v[cudaIndex3D( i, j-1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_D, node_velocity_w[cudaIndex3D( i, j-1, k, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_D,    node_density[cudaIndex3D( i, j-1, k, GNODES_X )] );

		if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_D );
		kernelZeroGrids __device_func__ ( velv_D );
		kernelZeroGrids __device_func__ ( velw_D );
		kernelZeroGrids __device_func__ ( dens_D );

		if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrFront not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_F, node_velocity_u[cudaIndex3D( i, j, k+1, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_F, node_velocity_v[cudaIndex3D( i, j, k+1, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_F, node_velocity_w[cudaIndex3D( i, j, k+1, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_F,    node_density[cudaIndex3D( i, j, k+1, GNODES_X )] );

		if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_F );
		kernelZeroGrids __device_func__ ( velv_F );
		kernelZeroGrids __device_func__ ( velw_F );
		kernelZeroGrids __device_func__ ( dens_F );

		if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	if ( ptr->ptrBack not_eq nullptr )
	{
		kernelCopyGrids __device_func__( velu_B, node_velocity_u[cudaIndex3D( i, j, k-1, GNODES_X )] );
		kernelCopyGrids __device_func__( velv_B, node_velocity_v[cudaIndex3D( i, j, k-1, GNODES_X )] );
		kernelCopyGrids __device_func__( velw_B, node_velocity_w[cudaIndex3D( i, j, k-1, GNODES_X )] );
		kernelCopyGrids __device_func__( dens_B,    node_density[cudaIndex3D( i, j, k-1, GNODES_X )] );

		if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
	else
	{
		kernelZeroGrids __device_func__ ( velu_B );
		kernelZeroGrids __device_func__ ( velv_B );
		kernelZeroGrids __device_func__ ( velw_B );
		kernelZeroGrids __device_func__ ( dens_B );

		if ( helper.GetCUDALastError( "device kernel: kernelZeroGrids failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}
};

void FluidSimProc::SaveNode( int i, int j, int k )
{
	helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );
	SimNode *ptr = gpu_node[cudaIndex3D( i, j, k, GNODES_X )];

	/* draw data back */
	kernelCopyGrids __device_func__( node_velocity_u[cudaIndex3D(i,j,k,GNODES_X)], velu_C );
	kernelCopyGrids __device_func__( node_velocity_v[cudaIndex3D(i,j,k,GNODES_X)], velv_C );
	kernelCopyGrids __device_func__( node_velocity_w[cudaIndex3D(i,j,k,GNODES_X)], velw_C );
	kernelCopyGrids __device_func__(    node_density[cudaIndex3D(i,j,k,GNODES_X)], dens_C );

	if ( helper.GetCUDALastError( "device kernel: kernelCopyGrids failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	/* draw volumetric data back */
	i += m_cursor.x;
	j += m_cursor.y;
	k += m_cursor.z;
	kernelPickData __device_func__( dev_visual, dev_den, i * GRIDS_X, j * GRIDS_X, k * GRIDS_X );

	if ( helper.GetCUDALastError( "device kernel: kernelPickData failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::AddSource( void )
{
	if ( decrease_times eqt 0 )
	{
		helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );
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

void FluidSimProc::InitBoundary( void )
{
	helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );

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

void FluidSimProc::VelocitySolver( void )
{
	// diffuse the velocity field (per axis):
	hostDiffusion( dev_u0, dev_u, VISOCITY, dev_obs, MACRO_VELOCITY_U );
	hostDiffusion( dev_v0, dev_v, VISOCITY, dev_obs, MACRO_VELOCITY_V );
	hostDiffusion( dev_w0, dev_w, VISOCITY, dev_obs, MACRO_VELOCITY_W );
	
	if ( helper.GetCUDALastError( "host function failed: hostDiffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );

	if ( helper.GetCUDALastError( "host function failed: hostProject", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	hostAdvection( dev_u0, dev_u, dev_obs, MACRO_VELOCITY_U, dev_u, dev_v, dev_w );
	hostAdvection( dev_v0, dev_v, dev_obs, MACRO_VELOCITY_V, dev_u, dev_v, dev_w );
	hostAdvection( dev_w0, dev_w, dev_obs, MACRO_VELOCITY_W, dev_u, dev_v, dev_w );

	if ( helper.GetCUDALastError( "host function failed: hostAdvection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

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

	if ( helper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::ZeroBuffers( void )
{
	helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );

	/* zero GPU buffer */
	for ( int i = 0; i < dev_buffers_num; i++ )
		kernelZeroGrids  __device_func__ ( dev_buffers[i] );

	/* zero host buffer */
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{
		kernelZeroGrids __device_func__ ( dev_density[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_u[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_v[i] );
		kernelZeroGrids __device_func__ ( dev_velocity_w[i] );

		if ( helper.GetCUDALastError( "device failed: kernelZeroGrids", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit( 1 );
		}
	}

	/* zero visual buffer */
	kernelZeroVolumetric __device_func__ ( dev_visual );
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );

	if ( helper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

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

	helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );
	kernelInteractNodes __device_func__
		( dens_C, dens_L, dens_R, dens_U, dens_D, dens_F, dens_B, left, right, up, down, front, back );
	kernelInteractNodes __device_func__
		( velu_C, velu_L, velu_R, velu_U, velu_D, velu_F, velu_B, left, right, up, down, front, back );
	kernelInteractNodes __device_func__
		( velv_C, velv_L, velv_R, velv_U, velv_D, velv_F, velv_B, left, right, up, down, front, back );
	kernelInteractNodes __device_func__
		( velw_C, velw_L, velw_R, velw_U, velw_D, velw_F, velw_B, left, right, up, down, front, back );
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
					LoadNode(i,j,k);					
					Interaction(i,j,k);					
					AddSource();					
					VelocitySolver();
					DensitySolver();					
					SaveNode(i,j,k);
					gpu_node[cudaIndex3D(i,j,k,GNODES_X)]->updated = true;
				}
			}
		}
	}
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	for ( int k = 0; k < CURSOR_X; k++ )
	{
		for ( int j = 0; j < CURSOR_X; j++ )
		{
			for ( int i = 0; i < CURSOR_X; i++ )
			{
				m_cursor.x = i;
				m_cursor.y = j;
				m_cursor.z = k;
				
				/* read host nodes */
				IO_ReadBuffers();
				
				/* solving NS equations */
				SolveNavierStokers();
				
				/* save updated nodes */
				IO_WriteBuffers();
			}
		}
	}

	/* finally, generate volumetric image */
	RefreshStatus( fluid );
};

void FluidSimProc::IO_ReadBuffers( void )
{
	int ni, nj, nk; ni = nj = nk = 0;

	helper.DeviceDim3D( &blockDim, &gridDim, 1024, 32, 64, 64, 64 );

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

void FluidSimProc::IO_WriteBuffers( void )
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