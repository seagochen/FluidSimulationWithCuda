/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 20, 2014
* <File Name>     FluidSimProc.cpp
*/

#include <iostream>
#include <cuda_runtime.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"

using namespace sge;

FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	if ( AllocateResource ( fluid ) != SG_RUNTIME_OK )
	{
		FreeResource ();
		exit (1);
	}

	/* build order */
	BuildOrder();

	/* select node */
	ActiveNode( 1, 0, 1 );

	/* clear buffer */
	ZeroBuffers();

	/* set boundary */
	InitBoundary( 1, 0, 1 );

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

void FluidSimProc::BuildOrder( void )
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

	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				printf ( "offset: %d %d %d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					host_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.x, 
					host_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.y, 
					host_node[cudaIndex3D( i, j, k, NODES_X )]->nodeIX.z,
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrLeft not_eq nullptr,
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrRight not_eq nullptr,
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrUp not_eq nullptr,
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrDown not_eq nullptr,
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrFront not_eq nullptr,
					host_node[cudaIndex3D( i, j, k, NODES_X )]->ptrBack not_eq nullptr );
			}
		}
	}
};

SGRUNTIMEMSG FluidSimProc::AllocateResource ( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		helper.CheckRuntimeErrors ( "cudaSetDevices", __FILE__, __LINE__ );

	/* allocate memory on host */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;

		/* simulation nodes */
		SimNode *node = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft = node->ptrRight = nullptr;
		node->ptrDown = node->ptrUp = nullptr;
		host_node.push_back( node );

		node->active = false;

		host_density.push_back( ptrDens );
		host_velocity_u.push_back( ptrU );
		host_velocity_v.push_back( ptrV );
		host_velocity_w.push_back( ptrW );
		host_obstacle.push_back( ptrObs );
	}

	/* allocate memory on GPU devices */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;

		dev_buffers.push_back(ptr);
	}

	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;

	/* allocate temporary buffers */
	if ( helper.CreateDeviceBuffers( sizeof(double)*TPBUFFER_X, 1, &dev_tpbufs ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;
	if ( helper.CreateHostBuffers(sizeof(double)*TPBUFFER_X, 1, &host_tpbufs ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;

	/* finally */
	return SG_RUNTIME_OK;
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
	}

	/* free device resource */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		helper.FreeDeviceBuffers( 1, &dev_buffers[i] );
	}

	/* free... */
	helper.FreeHostBuffers( 2, &host_visual, &host_tpbufs );
	helper.FreeDeviceBuffers( 2, &dev_visual, &dev_tpbufs );

	/* empty the vector */
	host_density.empty();
	host_velocity_u.empty();
	host_velocity_v.empty();
	host_velocity_w.empty();
	host_obstacle.empty();
	dev_buffers.empty( );
}

bool FluidSimProc::SelectNode( int i, int j, int k )
{
#if !TESTING_MODE
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		nPos.x = i;
		nPos.y = j;
		nPos.z = k;
		
		int ix = cudaIndex3D( i, j, k, NODES_X );
		return host_node[ix]->active;
	}

	return false;
#else
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		nPos.x = i;
		nPos.y = j;
		nPos.z = k;

		int ix = cudaIndex3D(i,j,k,NODES_X);
		int center = cudaIndex3D(1,0,1,NODES_X);
		int left   = cudaIndex3D(0,0,1,NODES_X);
		int right  = cudaIndex3D(2,0,1,NODES_X);
		int up     = cudaIndex3D(1,1,1,NODES_X);
		int front  = cudaIndex3D(1,0,2,NODES_X);
		int back   = cudaIndex3D(1,0,0,NODES_X);

		if ( ix eqt center or
			ix eqt left or
			ix eqt right or 
			ix eqt front or
			ix eqt back or
			ix eqt up )
			return true;
	}

	return false;
#endif
};

bool FluidSimProc::ActiveNode( int i, int j, int k )
{
	int ix;
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		ix = cudaIndex3D( i, j, k, NODES_X );
		host_node[ix]->active = true;
	}

	return host_node[ix]->active == true;
};

bool FluidSimProc::DeactiveNode( int i, int j, int k )
{
	int ix;
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		ix = cudaIndex3D( i, j, k, NODES_X );
		host_node[ix]->active = false;
	}

	return host_node[ix]->active == false;
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{

				/* select node */
				if ( SelectNode( i, j, k ) )
				{
					/* for fluid simulation, copy the data to device */
					NodetoDevice();
					
					/* Fluid process */
					AddSource();
					VelocitySolver();
					DensitySolver();
					
					/* retrieve data back to host */
					DevicetoNode();

					/* pick density */
					DensitytoVolumetric();
				}
			}
		}
	}
	/* finally, generate volumetric image */
	GetVolumetric( fluid );

	/* tracing */
	TracingDensity();
};

void FluidSimProc::GetVolumetric( FLUIDSPARAM *fluid )
{
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::NodetoDevice ( void )
{
	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	cudaMemcpy( dev_u, host_velocity_u[ix], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_v, host_velocity_v[ix], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_w, host_velocity_w[ix], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_den, host_density[ix], m_node_size, cudaMemcpyHostToDevice  );
	cudaMemcpy( dev_obs, host_obstacle[ix], m_node_size, cudaMemcpyHostToDevice );
};


void FluidSimProc::DevicetoNode ( void )
{
	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	cudaMemcpy( host_velocity_u[ix], dev_u, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_v[ix], dev_v, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_w[ix], dev_w, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_density[ix], dev_den, m_node_size, cudaMemcpyDeviceToHost  );

#if 0
	system("cls");
	printf( "no.1: %f \n", host_density[ix][Index(gst_header,gst_header,gst_header)] );
	printf( "no.2: %f \n", host_density[ix][Index(gst_tailer,gst_header,gst_header)] );
	printf( "no.3: %f \n", host_density[ix][Index(gst_tailer,gst_header,gst_tailer)] );
	printf( "no.4: %f \n", host_density[ix][Index(gst_header,gst_header,gst_tailer)] );
#endif
};