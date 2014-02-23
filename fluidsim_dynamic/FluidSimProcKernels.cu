/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 23, 2014
* <File Name>     FluidSimProcKernels.cu
*/

#include "FluidSimulationKernels.h"

using namespace sge;

__global__ void kernelZeroGrids( double *grid )
{
	GetIndex ();
	grid [ Index(i,j,k) ] = 0.f;
};

__global__ void kernelZeroVolumetric( SGUCHAR *visual )
{
	GetIndex();

	for ( int ii = 0; ii < NODES_X; ii++ )
	{
		for ( int jj = 0; jj < NODES_X; jj++ )
		{
			for ( int kk = 0; kk < NODES_X; kk++ )
			{
				int di = ii * GRIDS_X + i;
				int dj = jj * GRIDS_X + j;
				int dk = kk * GRIDS_X + k;
				
				/* zero data */
				visual[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;
			}
		}
	}
};

__global__ void kernelZeroTemporaryBuffers( double *bufs )
{
	GetIndex();

	bufs[threadIdx.x] = 0.f;
};

__global__ void kernelZeroTemporaryBuffers( int *bufs )
{
	bufs[threadIdx.x] = 0;
};

__global__ void kernelPickData
( unsigned char *data, const double *bufs, int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	double value = bufs[ Index(i, j, k) ];

	/* append data to volume data */
	int temp = atomicRound( value );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};

__host__ void hostPickData( SGUCHAR *data, const double *bufs, SGINT3 *nodeIX )
{
	cudaDeviceDim3D();

	nodeIX->x *= GRIDS_X;
	nodeIX->y *= GRIDS_X;
	nodeIX->z *= GRIDS_X;

	kernelPickData cudaDevice(gridDim, blockDim)
		( data, bufs, nodeIX->x, nodeIX->y, nodeIX->z );
};

__global__ void kernelCopyBuffer( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ void kernelSwapBuffer( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__host__ void hostSwapBuffer( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
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

	/* zero boundary buffers */
	kernelZeroGrids<<<gridDim, blockDim>>>( dev_obs );

	for ( int i = 0; i < host_obstacle.size(); i++ )
	{
		if ( cudaMemcpy( host_obstacle[i], dev_obs,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}
	}

	/* select middle node */
	SelectNode( i, j, k );

	const int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	/* set boundary */
	kernelSetBoundary<<<gridDim, blockDim>>>( dev_obs );
	
	if ( cudaMemcpy( host_obstacle[ix], dev_obs,
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
	hostSwapBuffer( dev_u0, dev_u );
	hostSwapBuffer( dev_v0, dev_v );
	hostSwapBuffer( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );
	
	// advect the velocity field (per axis):
	hostAdvection( dev_u0, dev_u, dev_obs, MACRO_VELOCITY_U, dev_u, dev_v, dev_w );
	hostAdvection( dev_v0, dev_v, dev_obs, MACRO_VELOCITY_V, dev_u, dev_v, dev_w );
	hostAdvection( dev_w0, dev_w, dev_obs, MACRO_VELOCITY_W, dev_u, dev_v, dev_w );
	hostSwapBuffer( dev_u0, dev_u );
	hostSwapBuffer( dev_v0, dev_v );
	hostSwapBuffer( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );
};

void FluidSimProc::DensitySolver( void )
{
	hostDiffusion( dev_den0, dev_den, DIFFUSION, dev_obs, MACRO_DENSITY );
	hostSwapBuffer( dev_den0, dev_den );
	hostAdvection ( dev_den, dev_den0, dev_obs, MACRO_DENSITY, dev_u, dev_v, dev_w );
};

void FluidSimProc::DensitytoVolumetric( void )
{
	hostPickData( dev_visual, dev_den, &nPos );
}

void FluidSimProc::ZeroBuffers( void )
{
	cudaDeviceDim3D();

	/* zero GPU buffer */
	for ( int i = 0; i < dev_buffers_num; i++ )
		kernelZeroGrids <<<gridDim, blockDim>>> ( dev_buffers[i] );

	/* zero host buffer */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		cudaMemcpy( host_density[i], dev_den, m_node_size, cudaMemcpyDeviceToHost);
		cudaMemcpy( host_velocity_u[i], dev_u, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_v, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_w, m_node_size, cudaMemcpyDeviceToHost );
	}

	/* zero visual buffer */
	kernelZeroVolumetric <<< gridDim, blockDim>>> ( dev_visual );

	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
};

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
		helper.GetCUDALastError ( "cudaSetDevices", __FILE__, __LINE__ );

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

__device__ void atomicHandleFaceLR( double *grids, double const *center, int header, int tailer )
{
	GetIndex();
	
	/* copy entire face from center node */
	grids[Index(header,j,k)] = center[Index(tailer,j,k)];

	/* wait, we have left something which may cause un-conservation of mass */
	grids[Index(header,sim_header,k)] = center[Index(tailer,sim_header,k)] / 2.f;
	grids[Index(header,sim_tailer,k)] = center[Index(tailer,sim_tailer,k)] / 2.f;
	grids[Index(header,j,sim_header)] = center[Index(tailer,j,sim_header)] / 2.f;
	grids[Index(header,j,sim_tailer)] = center[Index(tailer,j,sim_tailer)] / 2.f;

	grids[Index(header,sim_header,sim_header)] = center[Index(tailer,sim_header,sim_header)] / 3.f;
	grids[Index(header,sim_header,sim_tailer)] = center[Index(tailer,sim_header,sim_tailer)] / 3.f;
	grids[Index(header,sim_tailer,sim_header)] = center[Index(tailer,sim_tailer,sim_header)] / 3.f;
	grids[Index(header,sim_tailer,sim_tailer)] = center[Index(tailer,sim_tailer,sim_tailer)] / 3.f;
};
 
__device__ void atomicHandleFaceFB( double *grids, double const *center, int header, int tailer )
{
	GetIndex();

	grids[Index(i,header,k)] = center[Index(i,tailer,k)];

	grids[Index(sim_header,header,k)] = center[Index(sim_header,tailer,k)] / 2.f;
	grids[Index(sim_tailer,header,k)] = center[Index(sim_tailer,tailer,k)] / 2.f;
	grids[Index(i,header,sim_header)] = center[Index(i,tailer,sim_header)] / 2.f;
	grids[Index(i,header,sim_tailer)] = center[Index(i,tailer,sim_tailer)] / 2.f;

	grids[Index(sim_header,header,sim_header)] = center[Index(sim_header,tailer,sim_header)] / 2.f;
	grids[Index(sim_tailer,header,sim_header)] = center[Index(sim_tailer,tailer,sim_header)] / 2.f;
	grids[Index(sim_header,header,sim_tailer)] = center[Index(sim_header,tailer,sim_tailer)] / 2.f;
	grids[Index(sim_tailer,header,sim_tailer)] = center[Index(sim_tailer,tailer,sim_tailer)] / 2.f;
};

__device__ void atomicHandleFaceUD( double *grids, double const *center, int header, int tailer )
{
	GetIndex();
 
	grids[Index(i,j,header)] = center[Index(i,j,tailer)];
 
	grids[Index(sim_header,j,header)] = center[Index(sim_header,j,tailer)] / 2.f;
	grids[Index(i,sim_header,header)] = center[Index(i,sim_header,tailer)] / 2.f;
	grids[Index(sim_tailer,j,header)] = center[Index(sim_tailer,j,tailer)] / 2.f;
	grids[Index(i,sim_tailer,header)] = center[Index(i,sim_tailer,tailer)] / 2.f;

	grids[Index(sim_header,sim_header,header)] = center[Index(sim_header,sim_header,tailer)] / 3.f;
	grids[Index(sim_header,sim_tailer,header)] = center[Index(sim_header,sim_tailer,tailer)] / 3.f;
	grids[Index(sim_tailer,sim_header,header)] = center[Index(sim_tailer,sim_header,tailer)] / 3.f;
	grids[Index(sim_tailer,sim_tailer,header)] = center[Index(sim_tailer,sim_tailer,tailer)] / 3.f;
};


__global__ void kernelFloodingBuffers
	( double *left, double *right, double *up, double *down,
	double *front, double *back, double const *center )
{
	GetIndex();

	/* copying buffers from center grids */
	left [Index(sim_tailer,j,k)] = center[Index(sim_header,j,k)];
	right[Index(sim_header,j,k)] = center[Index(sim_tailer,j,k)];
	up   [Index(i,sim_header,k)] = center[Index(i,sim_tailer,k)];
	down [Index(i,sim_tailer,k)] = center[Index(i,sim_header,k)];
	back [Index(i,j,sim_tailer)] = center[Index(i,j,sim_header)];
	front[Index(i,j,sim_header)] = center[Index(i,j,sim_tailer)];
};

__global__ void kernelFloodBuffersBetweenNodes( double *grids, double *center, const int ops )
{
	switch ( ops )
 	{
	case MACRO_LEFT:
		atomicHandleFaceLR( grids, center, sim_tailer, sim_header );
		break;

	case MACRO_RIGHT:
		atomicHandleFaceLR( grids, center, sim_header, sim_tailer );
		break;

	case MACRO_UP:
		atomicHandleFaceUD( grids, center, sim_header, sim_tailer );
		break;

	case MACRO_DOWN:
		atomicHandleFaceUD( grids, center, sim_tailer, sim_header );
		break;

	case MACRO_FRONT:
		atomicHandleFaceFB( grids, center, sim_tailer, sim_header );
		break;

	case MACRO_BACK:
		atomicHandleFaceFB( grids, center, sim_header, sim_tailer );
		break;

	default:
		break;
	}

	GetIndex();

	center[Index(gst_header,j,k)] = grids[Index(gst_header,j,k)] = 0.f;
	center[Index(gst_tailer,j,k)] = grids[Index(gst_tailer,j,k)] = 0.f;
	center[Index(i,gst_header,k)] = grids[Index(i,gst_header,k)] = 0.f;
	center[Index(i,gst_tailer,k)] = grids[Index(i,gst_tailer,k)] = 0.f;
	center[Index(i,j,gst_header)] = grids[Index(i,j,gst_header)] = 0.f;
	center[Index(i,j,gst_tailer)] = grids[Index(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelSumDensity
	( double *buffs, 
	const double *center, const double *left, const double *right,
	const double *up,     const double *down, const double *front, const double *back )
{
#define num_dens_center  buffs[ 0 ]
#define num_dens_left    buffs[ 1 ]
#define num_dens_right   buffs[ 2 ]
#define num_dens_up      buffs[ 3 ]
#define num_dens_down    buffs[ 4 ]
#define num_dens_front   buffs[ 5 ]
#define num_dens_back    buffs[ 6 ]

	GetIndex();

	num_dens_left   += left[Index(i,j,k)];
	num_dens_right  += right[Index(i,j,k)];
	num_dens_up     += up[Index(i,j,k)];
	num_dens_down   += down[Index(i,j,k)];
	num_dens_front  += front[Index(i,j,k)];
	num_dens_back   += back[Index(i,j,k)];
	num_dens_center += center[Index(i,j,k)];
 	
#undef num_dens_center
#undef num_dens_left  
#undef num_dens_right 
#undef num_dens_up    
#undef num_dens_down  
#undef num_dens_front 
#undef num_dens_back 

};

__global__ void kernelFloodBoundary( double *grids )
{
	GetIndex();

	/* faces */
	grids[Index(gst_header,j,k)] = grids[Index(sim_header,j,k)];
	grids[Index(gst_tailer,j,k)] = grids[Index(sim_tailer,j,k)];
	grids[Index(i,gst_header,k)] = grids[Index(i,sim_header,k)];
	grids[Index(i,gst_tailer,k)] = grids[Index(i,sim_tailer,k)];
	grids[Index(i,j,gst_header)] = grids[Index(i,j,sim_header)];
	grids[Index(i,j,gst_tailer)] = grids[Index(i,j,sim_tailer)];

	/* edges */
	grids[Index(i,gst_header,gst_header)] = ( grids[Index(i,sim_header,gst_header)] + grids[Index(i,gst_header,sim_header)] ) / 2.f;
	grids[Index(i,gst_tailer,gst_header)] = ( grids[Index(i,sim_tailer,gst_header)] + grids[Index(i,gst_tailer,sim_header)] ) / 2.f;
	grids[Index(i,gst_header,gst_tailer)] = ( grids[Index(i,sim_header,gst_tailer)] + grids[Index(i,gst_header,sim_tailer)] ) / 2.f;
	grids[Index(i,gst_tailer,gst_tailer)] = ( grids[Index(i,sim_tailer,gst_tailer)] + grids[Index(i,gst_tailer,sim_tailer)] ) / 2.f;

	grids[Index(gst_header,j,gst_header)] = ( grids[Index(sim_header,j,gst_header)] + grids[Index(gst_header,j,sim_header)] ) / 2.f;
	grids[Index(gst_tailer,j,gst_header)] = ( grids[Index(sim_tailer,j,gst_header)] + grids[Index(gst_tailer,j,sim_header)] ) / 2.f;
	grids[Index(gst_header,j,gst_tailer)] = ( grids[Index(sim_header,j,gst_tailer)] + grids[Index(gst_header,j,sim_tailer)] ) / 2.f;
	grids[Index(gst_tailer,j,gst_tailer)] = ( grids[Index(sim_tailer,j,gst_tailer)] + grids[Index(gst_tailer,j,sim_tailer)] ) / 2.f;

	grids[Index(gst_header,gst_header,k)] = ( grids[Index(sim_header,gst_header,k)] + grids[Index(gst_header,sim_header,k)] ) / 2.f;
	grids[Index(gst_tailer,gst_header,k)] = ( grids[Index(sim_tailer,gst_header,k)] + grids[Index(gst_tailer,sim_header,k)] ) / 2.f;
	grids[Index(gst_header,gst_tailer,k)] = ( grids[Index(sim_header,gst_tailer,k)] + grids[Index(gst_header,sim_tailer,k)] ) / 2.f;
	grids[Index(gst_tailer,gst_tailer,k)] = ( grids[Index(sim_tailer,gst_tailer,k)] + grids[Index(gst_tailer,sim_tailer,k)] ) / 2.f;

	/* vetices */
	grids[Index(gst_header,gst_header,gst_header)] = ( grids[Index(sim_header,gst_header,gst_header)] + grids[Index(gst_header,sim_header,gst_header)] + grids[Index(gst_header,gst_header,sim_header)] ) / 3.f;
	grids[Index(gst_header,gst_header,gst_tailer)] = ( grids[Index(sim_header,gst_header,gst_tailer)] + grids[Index(gst_header,sim_header,gst_tailer)] + grids[Index(gst_header,gst_header,sim_tailer)] ) / 3.f;
	grids[Index(gst_header,gst_tailer,gst_header)] = ( grids[Index(sim_header,gst_tailer,gst_header)] + grids[Index(gst_header,sim_tailer,gst_header)] + grids[Index(gst_header,gst_tailer,sim_header)] ) / 3.f;
	grids[Index(gst_header,gst_tailer,gst_tailer)] = ( grids[Index(sim_header,gst_tailer,gst_tailer)] + grids[Index(gst_header,sim_tailer,gst_tailer)] + grids[Index(gst_header,gst_tailer,sim_tailer)] ) / 3.f;
	grids[Index(gst_tailer,gst_header,gst_header)] = ( grids[Index(sim_tailer,gst_header,gst_header)] + grids[Index(gst_tailer,sim_header,gst_header)] + grids[Index(gst_tailer,gst_header,sim_header)] ) / 3.f;
	grids[Index(gst_tailer,gst_header,gst_tailer)] = ( grids[Index(sim_tailer,gst_header,gst_tailer)] + grids[Index(gst_tailer,sim_header,gst_tailer)] + grids[Index(gst_tailer,gst_header,sim_tailer)] ) / 3.f;
	grids[Index(gst_tailer,gst_tailer,gst_header)] = ( grids[Index(sim_tailer,gst_tailer,gst_header)] + grids[Index(gst_tailer,sim_tailer,gst_header)] + grids[Index(gst_tailer,gst_tailer,sim_header)] ) / 3.f;
	grids[Index(gst_tailer,gst_tailer,gst_tailer)] = ( grids[Index(sim_tailer,gst_tailer,gst_tailer)] + grids[Index(gst_tailer,sim_tailer,gst_tailer)] + grids[Index(gst_tailer,gst_tailer,sim_tailer)]) / 3.f;
};



void FluidSimProc::TracingDensity( void )
{
	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				if ( SelectNode( i, j, k ) )
				{
					DataFlooding( host_density,    i, j, k, true );
					DataFlooding( host_velocity_u, i, j, k, false );
					DataFlooding( host_velocity_v, i, j, k, false );
					DataFlooding( host_velocity_w, i, j, k, false );
				}
			}
		}
	}
};

void FluidSimProc::UploadNeighbouringBuffers( vector<double*> container, int i, int j, int k )
{
	/* navigated current node from node list */
	SimNode *node = host_node[cudaIndex3D( i, j, k, NODES_X )];
		
	/* upload center buffers */
	if ( cudaMemcpy( dev_center, container[cudaIndex3D(i,j,k,NODES_X)],
		m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
	{
		helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

	/* upload left buffers */
	if ( node->ptrLeft not_eq nullptr )
	{
		if ( cudaMemcpy( dev_left, container[cudaIndex3D(i-1,j,k,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* upload right buffers */
	if ( node->ptrRight not_eq nullptr )
	{
		if ( cudaMemcpy( dev_right, container[cudaIndex3D(i+1,j,k,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* upload up buffers */
	if ( node->ptrUp not_eq nullptr )
	{
		if ( cudaMemcpy( dev_up, container[cudaIndex3D(i,j+1,k,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* upload down buffers */
	if ( node->ptrDown not_eq nullptr )
	{
		if ( cudaMemcpy( dev_down, container[cudaIndex3D(i,j-1,k,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* upload front buffers */
	if ( node->ptrFront not_eq nullptr )
	{
		if ( cudaMemcpy( dev_front, container[cudaIndex3D(i,j,k+1,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* upload back buffers */
	if ( node->ptrBack not_eq nullptr )
	{
		if ( cudaMemcpy( dev_back, container[cudaIndex3D(i,j,k-1,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}
};

void FluidSimProc::DownloadNeighbouringBuffers( vector<double*> container, int i, int j, int k )
{
	/* navigated current node from node list */
	SimNode *node = host_node[cudaIndex3D( i, j, k, NODES_X )];
		
	/* download center buffers */
	if ( cudaMemcpy( container[cudaIndex3D(i,j,k,NODES_X)], dev_center,
		m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

	/* download left buffers */
	if ( node->ptrLeft not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i-1,j,k,NODES_X)], dev_left, 
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* download right buffers */
	if ( node->ptrRight not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i+1,j,k,NODES_X)], dev_right,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* download up buffers */
	if ( node->ptrUp not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i,j+1,k,NODES_X)], dev_up,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* download down buffers */
	if ( node->ptrDown not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i,j-1,k,NODES_X)], dev_down,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* download front buffers */
	if ( node->ptrFront not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i,j,k+1,NODES_X)], dev_front,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}

	/* download back buffers */
	if ( node->ptrBack not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i,j,k-1,NODES_X)], dev_back,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}
	}
};

void FluidSimProc::DataFlooding( vector<double*> container, int i, int j, int k, bool isDensity )
{
	/* upload neighbouring data to GPU buffers */
	UploadNeighbouringBuffers( container, i, j, k );

	/* flooding neighbouring buffers*/
	cudaDeviceDim3D();

	kernelFloodingBuffers <<<gridDim,blockDim>>> 
		( dev_left, dev_right, dev_up, dev_down, dev_front, dev_back, dev_center );
/*	
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_left, dev_center, MACRO_LEFT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_right, dev_center, MACRO_RIGHT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_up, dev_center, MACRO_UP );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_down, dev_center, MACRO_DOWN );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_front, dev_center, MACRO_FRONT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_back, dev_center, MACRO_BACK );
*/
	if ( isDensity )
	{
		/* zero temporary buffers */
		kernelZeroTemporaryBuffers <<<1, TPBUFFER_X>>> ( dev_tpbufs );

		/* check */
		kernelSumDensity<<<gridDim, blockDim>>> 
			( dev_tpbufs, dev_center, dev_left, dev_right, dev_up, dev_down, dev_front, dev_back );

		if ( cudaMemcpy( host_tpbufs, dev_tpbufs, sizeof(double) * TPBUFFER_X, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}

#if TESTING_MODE
		/* navigated current node from node list */
		SimNode *node = host_node[cudaIndex3D( i, j, k, NODES_X )];

		system( "cls" );
		printf( "%d %d %d || left: %d right %d up %d down %d front %d back %d\n",
			node->nodeIX.x, node->nodeIX.y, node->nodeIX.z, 
			node->ptrLeft  not_eq nullptr,
			node->ptrRight not_eq nullptr,
			node->ptrUp    not_eq nullptr,
			node->ptrDown  not_eq nullptr,
			node->ptrFront not_eq nullptr,
			node->ptrBack  not_eq nullptr );
		printf( "center:  %f\n", host_tpbufs[ 0 ] );
		printf( "left:    %f\n", host_tpbufs[ 1 ] );
		printf( "right:   %f\n", host_tpbufs[ 2 ] );
		printf( "up:      %f\n", host_tpbufs[ 3 ] );
		printf( "down:    %f\n", host_tpbufs[ 4 ] );
		printf( "front:   %f\n", host_tpbufs[ 5 ] );
		printf( "back:    %f\n", host_tpbufs[ 6 ] );
#endif

	}	
	/* retrieve data back to node */
	DownloadNeighbouringBuffers( container, i, j, k );
};