/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 01, 2014
* <File Name>     FluidSimProcKernels.cu
*/

#include <iostream>
#include <utility>
#include "FluidSimProc.h"
#include "CUDACustomMath.h"

using namespace sge;

#pragma region CUDA kernel function

__global__ void kernelCopyGrids( double *dst, const double *src )
{
	GetIndex3D();

	dst[ Index3D(i,j,k,GRID_X) ] = src[ Index3D(i,j,k,GRID_X) ];
};

__global__ void kernelZeroGrids( double *grid )
{
	GetIndex3D ();
	grid [  Index3D(i,j,k,GRID_X)  ] = 0.f;
};

__global__ void kernelZeroVolumetric( SGUCHAR *visual )
{
	GetIndex3D();

	for ( int ii = 0; ii < NODE_X; ii++ )
	{
		for ( int jj = 0; jj < NODE_X; jj++ )
		{
			for ( int kk = 0; kk < NODE_X; kk++ )
			{
				int di = ii * GRID_X + i;
				int dj = jj * GRID_X + j;
				int dk = kk * GRID_X + k;
				
				/* zero data */
				visual[ Index3D(di, dj, dk, VOLUME_X) ] = 0;
			}
		}
	}
};

__global__ void kernelZeroTemporaryBuffers( double *bufs )
{
	GetIndex3D();

	bufs[threadIdx.x] = 0.f;
};

__global__ void kernelZeroTemporaryBuffers( int *bufs )
{
	bufs[threadIdx.x] = 0;
};

__global__ void kernelPickData
( SGUCHAR *data, const double *bufs, const int offseti, const int offsetj, const int offsetk )
{
	GetIndex3D();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data[ Index3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	double value = bufs[ Index3D(i, j, k, GRID_X) ];

	/* append data to volume data */
	int temp = atomicRound( value );
	if ( temp > 0 and temp < 250 )
		data [ Index3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};

__global__ void kernelSlotLoad
	( double *slot, double const *grid, double const *grid_U, double const *grid_D,
	double const *grid_L, double const *grid_R, double const *grid_F, double const *grid_B )
{
	GetIndex3D();

	int si = i + 1;
	int sj = j + 1;
	int sk = k + 1;
	const int g_header = 0;
	const int g_tailer = 65;
	const int s_header = 1;
	const int s_tailer = 64;

	slot[ Index3D(si, sj, sk, SLOT_X) ] = grid[  Index3D(i,j,k,GRID_X)  ];

	slot[ Index3D(g_header, sj, sk, SLOT_X) ] = grid_L[ Index3D(63, j, k, GRID_X) ];
	slot[ Index3D(g_tailer, sj, sk, SLOT_X) ] = grid_R[ Index3D(0, j, k, GRID_X) ];
	slot[ Index3D(si, g_header, sk, SLOT_X) ] = grid_D[ Index3D(i, 63, k, GRID_X) ];
	slot[ Index3D(si, g_tailer, sk, SLOT_X) ] = grid_U[ Index3D(i, 0, k, GRID_X) ];
	slot[ Index3D(si, sj, g_header, SLOT_X) ] = grid_B[ Index3D(i, j, 63, GRID_X) ];
	slot[ Index3D(si, sj, g_tailer, SLOT_X) ] = grid_F[ Index3D(i, j, 0, GRID_X) ];

	slot[ Index3D(g_header, g_header, sk, SLOT_X) ] = ( slot[ Index3D(s_header, g_header, sk, SLOT_X) ] + slot[ Index3D(g_header, s_header, sk, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_header, g_tailer, sk, SLOT_X) ] = ( slot[ Index3D(s_header, g_tailer, sk, SLOT_X) ] + slot[ Index3D(g_header, s_tailer, sk, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_tailer, g_header, sk, SLOT_X) ] = ( slot[ Index3D(s_tailer, g_header, sk, SLOT_X) ] + slot[ Index3D(g_tailer, s_header, sk, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_tailer, g_tailer, sk, SLOT_X) ] = ( slot[ Index3D(s_tailer, g_tailer, sk, SLOT_X) ] + slot[ Index3D(g_tailer, s_tailer, sk, SLOT_X) ] ) / 2.f;
	slot[ Index3D(si, g_header, g_header, SLOT_X) ] = ( slot[ Index3D(si, s_header, g_header, SLOT_X) ] + slot[ Index3D(si, g_header, s_header, SLOT_X) ] ) / 2.f;
	slot[ Index3D(si, g_header, g_tailer, SLOT_X) ] = ( slot[ Index3D(si, s_header, g_tailer, SLOT_X) ] + slot[ Index3D(si, g_header, s_tailer, SLOT_X) ] ) / 2.f;
	slot[ Index3D(si, g_tailer, g_header, SLOT_X) ] = ( slot[ Index3D(si, s_tailer, g_header, SLOT_X) ] + slot[ Index3D(si, g_tailer, s_header, SLOT_X) ] ) / 2.f;
	slot[ Index3D(si, g_tailer, g_tailer, SLOT_X) ] = ( slot[ Index3D(si, s_tailer, g_tailer, SLOT_X) ] + slot[ Index3D(si, g_tailer, s_tailer, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_header, sj, g_header, SLOT_X) ] = ( slot[ Index3D(s_header, sj, g_header, SLOT_X) ] + slot[ Index3D(g_header, sj, s_header, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_header, sj, g_tailer, SLOT_X) ] = ( slot[ Index3D(s_header, sj, g_tailer, SLOT_X) ] + slot[ Index3D(g_header, sj, s_tailer, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_tailer, sj, g_header, SLOT_X) ] = ( slot[ Index3D(s_tailer, sj, g_header, SLOT_X) ] + slot[ Index3D(g_tailer, sj, s_header, SLOT_X) ] ) / 2.f;
	slot[ Index3D(g_tailer, sj, g_tailer, SLOT_X) ] = ( slot[ Index3D(s_tailer, sj, g_tailer, SLOT_X) ] + slot[ Index3D(g_tailer, sj, s_tailer, SLOT_X) ] ) / 2.f;

	slot[ Index3D(g_header, g_header, g_header, SLOT_X) ] = ( slot[ Index3D(g_header, s_header, g_header, SLOT_X) ] + slot[ Index3D(s_header, g_header, g_header, SLOT_X) ] + slot[ Index3D(g_header, g_header, s_header, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_header, g_header, g_tailer, SLOT_X) ] = ( slot[ Index3D(g_header, s_header, g_tailer, SLOT_X) ] + slot[ Index3D(s_header, g_header, g_tailer, SLOT_X) ] + slot[ Index3D(g_header, g_header, s_tailer, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_header, g_tailer, g_header, SLOT_X) ] = ( slot[ Index3D(g_header, s_tailer, g_header, SLOT_X) ] + slot[ Index3D(s_header, g_tailer, g_header, SLOT_X) ] + slot[ Index3D(g_header, g_tailer, s_header, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_header, g_tailer, g_tailer, SLOT_X) ] = ( slot[ Index3D(g_header, s_tailer, g_tailer, SLOT_X) ] + slot[ Index3D(s_header, g_tailer, g_tailer, SLOT_X) ] + slot[ Index3D(g_header, g_tailer, s_tailer, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_tailer, g_header, g_header, SLOT_X) ] = ( slot[ Index3D(g_tailer, s_header, g_header, SLOT_X) ] + slot[ Index3D(s_tailer, g_header, g_header, SLOT_X) ] + slot[ Index3D(g_tailer, g_header, s_header, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_tailer, g_header, g_tailer, SLOT_X) ] = ( slot[ Index3D(g_tailer, s_header, g_tailer, SLOT_X) ] + slot[ Index3D(s_tailer, g_header, g_tailer, SLOT_X) ] + slot[ Index3D(g_tailer, g_header, s_tailer, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_tailer, g_tailer, g_header, SLOT_X) ] = ( slot[ Index3D(g_tailer, s_tailer, g_header, SLOT_X) ] + slot[ Index3D(s_tailer, g_tailer, g_header, SLOT_X) ] + slot[ Index3D(g_tailer, g_tailer, s_header, SLOT_X) ] ) / 3.f;
	slot[ Index3D(g_tailer, g_tailer, g_tailer, SLOT_X) ] = ( slot[ Index3D(g_tailer, s_tailer, g_tailer, SLOT_X) ] + slot[ Index3D(s_tailer, g_tailer, g_tailer, SLOT_X) ] + slot[ Index3D(g_tailer, g_tailer, s_tailer, SLOT_X) ] ) / 3.f;
}

__global__ void kernelSlotLoad( double *slot, double const *grid )
{
	GetIndex3D();

	int si = i + 1;
	int sj = j + 1;
	int sk = k + 1;

	slot[ Index3D(si, sj, sk, SLOT_X) ] = grid[ Index3D(i,j,k,GRID_X) ];
};

__global__ void kernelSlotExit( double *grid, double const *slot ) 
{
	GetIndex3D();

	int si = i + 1;
	int sj = j + 1;
	int sk = k + 1;

	grid[  Index3D(i,j,k,GRID_X)  ] = slot[ Index3D(si,sj,sk,SLOT_X) ];
};

#pragma endregion


FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
{
	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	if ( !AllocateResource( fluid ) ) { FreeResource(); exit (1); }

	/* build order */
	BuildOrder();

	/* clear buffer */
	ZeroBuffers();

	/* select node */
	ActiveTheNode( 1, 0, 1 );

	/* set boundary */
	InitBoundary( 1, 0, 1 );

	/* upload buffers */
	UploadBuffers();

	/* finally, print message */
	printf( "fluid simulation ready...\n" );
};

void FluidSimProc::UploadBuffers( void )
{
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		cudaMemcpy( dev_densS[i], host_dens[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_veluS[i], host_velu[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velvS[i], host_velv[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_velwS[i], host_velw[i], m_node_size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_obst[i],  host_obst[i], m_node_size, cudaMemcpyHostToDevice );

		if ( helper.GetCUDALastError( "cudaMemcpy failed", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit(1);
		}
	}
};

void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	fluid->fps.dwCurrentTime = 0;
	fluid->fps.dwElapsedTime = 0;
	fluid->fps.dwFrames = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS = 0;

	m_node_size = GRID_X * GRID_X * GRID_X * sizeof(double);
	m_volm_size = VOLUME_X * VOLUME_X * VOLUME_X * sizeof(SGUCHAR);
	m_slot_size = SLOT_X * SLOT_X * SLOT_X * sizeof(double);

	increase_times = decrease_times = 0;
	dTimes = 0.f;
};

bool FluidSimProc::AllocateResource( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
	{
		helper.GetCUDALastError ( "cudaSetDevices", __FILE__, __LINE__ );
		return false;
	}

	/* allocate computing nodes in memory */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;		

		host_dens.push_back( ptrDens );
		host_velu.push_back( ptrU );
		host_velv.push_back( ptrV );
		host_velw.push_back( ptrW );
		host_obst.push_back( ptrObs );
	}

	/* create node's mark */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		SimNode *node = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft = node->ptrRight = nullptr;
		node->ptrDown = node->ptrUp = nullptr;
		node->active = false;
		host_node.push_back( node );
	}

	/* allocate buffers of node in CUDA device */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		double *ptrDenS, *ptrUS, *ptrVS, *ptrWS, *ptrObs;
		double *ptrDenT, *ptrUT, *ptrVT, *ptrWT;

		if ( helper.CreateDeviceBuffers( m_node_size, 2, &ptrDenS, &ptrDenT ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 2, &ptrUS, &ptrUT ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 2, &ptrVS, &ptrVT ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 2, &ptrWS, &ptrWT ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK ) return false;

		/* push to S */
		dev_densS.push_back( ptrDenS );
		dev_veluS.push_back( ptrUS );
		dev_velvS.push_back( ptrVS );
		dev_velwS.push_back( ptrWS );

		/* push to T */
		dev_densT.push_back( ptrDenT );
		dev_veluT.push_back( ptrUT );
		dev_velvT.push_back( ptrVT );
		dev_velwT.push_back( ptrWT );

		/* etc. */
		dev_obst.push_back( ptrObs );
	}

	/* allocate slot in GPU device */
	for ( int i = 0; i < dev_slot_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_slot_size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;
		dev_slot.push_back(ptr);
	}

	/* allocate temporary buffers in GPU device */
	for ( int i = 0; i < dev_bufs_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;
		dev_bufs.push_back(ptr);
	}

	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK ) return false;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )  return false;

	/* finally */
	return true;
}  

void FluidSimProc::FreeResource( void )
{
	/* free resource */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		helper.FreeHostBuffers( 5, &host_dens[i], &host_velu[i], &host_velv[i], &host_velw[i], &host_obst[i] );
		helper.FreeDeviceBuffers( 4, &dev_densS[i], &dev_veluS[i], &dev_velvS[i], &dev_velwS[i] );
		helper.FreeDeviceBuffers( 4, &dev_densT[i], &dev_veluT[i], &dev_velvT[i], &dev_velwT[i] );
		helper.FreeDeviceBuffers( 1, &dev_obst[i] );
	}

	/* free slot buffers */
	for ( int i = 0; i < dev_slot_num; i++ )
		helper.FreeDeviceBuffers( 1, &dev_slot[i] );
	for ( int i = 0; i < dev_bufs_num; i++ )
		helper.FreeDeviceBuffers( 1, &dev_bufs[i] );

	/* free... */
	helper.FreeHostBuffers( 1, &host_visual );
	helper.FreeDeviceBuffers( 1, &dev_visual );

}

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	for ( int i = 0; i < NODE_X; i++ )
	{
		for ( int j = 0; j < NODE_X; j++ )
		{
			for ( int k = 0; k < NODE_X; k++ )
			{
				/* upload origin grid to slot for fluid simulation */
				LoadBullet( i, j, k );
				
				/* Fluid process */
				AddSource();
				VelocitySolver();
				DensitySolver();

				/* retrieve data back to host */
				ExitBullet( i, j, k );
			}
		}
	}
	/* finally, generate volumetric image */
	GetVolumetric( fluid );
};

void FluidSimProc::LoadBullet( int i, int j, int k )
{
	cudaDeviceDim3D();

	for ( int i = 0; i < dev_bufs_num; i++ )
		kernelZeroGrids <<<gridDim, blockDim>>> ( dev_bufs[i] );

	/* copy center grid from node to temporary buffers */
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_C, dev_densS[Index3D(i,j,k,NODE_X)] );
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_C, dev_veluS[Index3D(i,j,k,NODE_X)] );
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_C, dev_velvS[Index3D(i,j,k,NODE_X)] );
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_C, dev_velwS[Index3D(i,j,k,NODE_X)] );

	/* copy neighbouring */
	if ( host_node[Index3D(i,j,k,NODE_X)]->ptrLeft not_eq nullptr )
	{
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_L, dev_densS[Index3D(i-1,j,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_L, dev_veluS[Index3D(i-1,j,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_L, dev_velvS[Index3D(i-1,j,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_L, dev_velwS[Index3D(i-1,j,k,NODE_X)] );
	}
	if ( host_node[Index3D(i,j,k,NODE_X)]->ptrRight not_eq nullptr )
	{
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_R, dev_densS[Index3D(i+1,j,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_R, dev_veluS[Index3D(i+1,j,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_R, dev_velvS[Index3D(i+1,j,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_R, dev_velwS[Index3D(i+1,j,k,NODE_X)] );
	}
	if ( host_node[Index3D(i,j,k,NODE_X)]->ptrUp not_eq nullptr )
	{
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_U, dev_densS[Index3D(i,j+1,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_U, dev_veluS[Index3D(i,j+1,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_U, dev_velvS[Index3D(i,j+1,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_U, dev_velwS[Index3D(i,j+1,k,NODE_X)] );
	}
	if ( host_node[Index3D(i,j,k,NODE_X)]->ptrDown not_eq nullptr )
	{
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_D, dev_densS[Index3D(i,j-1,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_D, dev_veluS[Index3D(i,j-1,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_D, dev_velvS[Index3D(i,j-1,k,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_D, dev_velwS[Index3D(i,j-1,k,NODE_X)] );
	}
	if ( host_node[Index3D(i,j,k,NODE_X)]->ptrFront not_eq nullptr )
	{
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_F, dev_densS[Index3D(i,j,k+1,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_F, dev_veluS[Index3D(i,j,k+1,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_F, dev_velvS[Index3D(i,j,k+1,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_F, dev_velwS[Index3D(i,j,k+1,NODE_X)] );
	}
	if ( host_node[Index3D(i,j,k,NODE_X)]->ptrBack not_eq nullptr )
	{
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_dens_B, dev_densS[Index3D(i,j,k-1,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velu_B, dev_veluS[Index3D(i,j,k-1,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velv_B, dev_velvS[Index3D(i,j,k-1,NODE_X)] );
		kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velw_B, dev_velwS[Index3D(i,j,k-1,NODE_X)] );
	}

	/* load bullet */
	kernelSlotLoad <<<gridDim, blockDim>>> ( dev_slot_d, dev_dens_C, dev_dens_U, dev_dens_D, dev_dens_L, dev_dens_R, dev_dens_F, dev_dens_B );
	kernelSlotLoad <<<gridDim, blockDim>>> ( dev_slot_u, dev_velu_C, dev_velu_U, dev_velu_D, dev_velu_L, dev_velu_R, dev_velu_F, dev_velu_B );
	kernelSlotLoad <<<gridDim, blockDim>>> ( dev_slot_v, dev_velv_C, dev_velv_U, dev_velv_D, dev_velv_L, dev_velv_R, dev_velv_F, dev_velv_B );
	kernelSlotLoad <<<gridDim, blockDim>>> ( dev_slot_w, dev_velw_C, dev_velw_U, dev_velw_D, dev_velw_L, dev_velw_R, dev_velw_F, dev_velw_B );
	kernelSlotLoad <<<gridDim, blockDim>>> ( dev_slot_obs, dev_obst[Index3D(i,j,k,NODE_X)] );
};

void FluidSimProc::ExitBullet( int i, int j, int k )
{
	cudaDeviceDim3D();

	/* load bullet */
	kernelSlotExit <<<gridDim, blockDim>>> ( dev_dens_C, dev_slot_d );
	kernelSlotExit <<<gridDim, blockDim>>> ( dev_velu_C, dev_slot_u );
	kernelSlotExit <<<gridDim, blockDim>>> ( dev_velv_C, dev_slot_v );
	kernelSlotExit <<<gridDim, blockDim>>> ( dev_velw_C, dev_slot_w );

	/* save data */
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_densT[Index3D(i,j,k,NODE_X)], dev_dens_C );
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_veluT[Index3D(i,j,k,NODE_X)], dev_velu_C );
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velvT[Index3D(i,j,k,NODE_X)], dev_velv_C );
	kernelCopyGrids <<<gridDim, blockDim>>> ( dev_velwT[Index3D(i,j,k,NODE_X)], dev_velw_C );

	/* draw volumetric data back */
	kernelPickData <<<gridDim, blockDim>>>
		( dev_visual, dev_dens_C, i * GRID_X, j * GRID_X, k * GRID_X );
};

void FluidSimProc::GetVolumetric( FLUIDSPARAM *fluid )
{
	/* swap original and updated buffers */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		std::swap( dev_densS[i], dev_densT[i] );
		std::swap( dev_veluS[i], dev_veluT[i] );
		std::swap( dev_velvS[i], dev_velvT[i] );
		std::swap( dev_velwS[i], dev_velwT[i] );
	}

	/* retrieve and rendering data */
	if ( cudaThreadSynchronize() not_eq cudaSuccess )
	{
		printf( "cudaThreadSynchronize failed\n" );
		FreeResource();
		exit( 1 );
	}

	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::AddSource( void )
{

};

void FluidSimProc::InitBoundary( int i, int j, int k )
{

};

void FluidSimProc::VelocitySolver( void )
{

};

void FluidSimProc::DensitySolver( void )
{

};

void FluidSimProc::ZeroBuffers( void )
{
	cudaDeviceDim3D();

	/* zero buffers */
	for ( int i = 0; i < dev_bufs_num; i++ ) kernelZeroGrids<<<gridDim, blockDim>>> ( dev_bufs[i] );

	/* zero host buffer */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		cudaMemcpy( host_dens[i], dev_dens_C, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velu[i], dev_dens_C, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velv[i], dev_dens_C, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velw[i], dev_dens_C, m_node_size, cudaMemcpyDeviceToHost );

		if ( helper.GetCUDALastError( "cudaMemcpy failed when zero buffers", __FILE__, __LINE__ ) )
		{
			FreeResource();
			exit(1);
		}
	}

	/* zero GPU buffer */
	for ( int i = 0; i < NODE_X * NODE_X * NODE_X; i++ )
	{
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_densS[i] );
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_veluS[i] );
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_velvS[i] );
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_velwS[i] );

		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_densT[i] );
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_veluT[i] );
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_velvT[i] );
		 kernelZeroGrids<<<gridDim, blockDim>>> ( dev_velwT[i] );
	}
	
	/* zero visual buffer */
	kernelZeroVolumetric <<< gridDim, blockDim>>> ( dev_visual );
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
	
	if ( helper.GetCUDALastError( "cudaMemcpy failed when zero buffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

#pragma region obsolete methods

void FluidSimProc::BuildOrder( void )
{
	for ( int k = 0; k < NODE_X; k++ )
	{
		for ( int j = 0; j < NODE_X; j++ )
		{
			for ( int i = 0; i < NODE_X; i++ )
			{
				/* left */
				if ( i >= 1 )
					host_node[Index3D( i, j, k, NODE_X )]->ptrLeft  = host_node[Index3D( i-1, j, k, NODE_X )];
				/* right */
				if ( i <= NODE_X - 2 )
					host_node[Index3D( i, j, k, NODE_X )]->ptrRight = host_node[Index3D( i+1, j, k, NODE_X )];
				/* down */
				if ( j >= 1 )
					host_node[Index3D( i, j, k, NODE_X )]->ptrDown  = host_node[Index3D( i, j-1, k, NODE_X )];
				/* up */
				if ( j <= NODE_X - 2 )
					host_node[Index3D( i, j, k, NODE_X )]->ptrUp    = host_node[Index3D( i, j+1, k, NODE_X )];
				/* back */
				if ( k >= 1 )
					host_node[Index3D( i, j, k, NODE_X )]->ptrBack  = host_node[Index3D( i, j, k-1, NODE_X )];
				/* front */
				if ( k <= NODE_X - 2 )
					host_node[Index3D( i, j, k, NODE_X )]->ptrFront = host_node[Index3D( i, j, k+1, NODE_X )];

				host_node[Index3D( i, j, k, NODE_X )]->nodeIX.x = i;
				host_node[Index3D( i, j, k, NODE_X )]->nodeIX.y = j;
				host_node[Index3D( i, j, k, NODE_X )]->nodeIX.z = k;
			}
		}
	}

	for ( int i = 0; i < NODE_X; i++ )
	{
		for ( int j = 0; j < NODE_X; j++ )
		{
			for ( int k = 0; k < NODE_X; k++ )
			{
				printf ( "offset: %d %d %d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					host_node[Index3D( i, j, k, NODE_X )]->nodeIX.x, 
					host_node[Index3D( i, j, k, NODE_X )]->nodeIX.y, 
					host_node[Index3D( i, j, k, NODE_X )]->nodeIX.z,
					host_node[Index3D( i, j, k, NODE_X )]->ptrLeft not_eq nullptr,
					host_node[Index3D( i, j, k, NODE_X )]->ptrRight not_eq nullptr,
					host_node[Index3D( i, j, k, NODE_X )]->ptrUp not_eq nullptr,
					host_node[Index3D( i, j, k, NODE_X )]->ptrDown not_eq nullptr,
					host_node[Index3D( i, j, k, NODE_X )]->ptrFront not_eq nullptr,
					host_node[Index3D( i, j, k, NODE_X )]->ptrBack not_eq nullptr );
			}
		}
	}
};

bool FluidSimProc::ActiveTheNode( int i, int j, int k )
{
	int ix;
	if ( i >= 0 and i < NODE_X and j >= 0 and j < NODE_X and k >= 0 and k < NODE_X )
	{
		ix = Index3D( i, j, k, NODE_X );
		host_node[ix]->active = true;
	}

	return host_node[ix]->active == true;
};

bool FluidSimProc::DeactiveTheNode( int i, int j, int k )
{
	int ix;
	if ( i >= 0 and i < NODE_X and j >= 0 and j < NODE_X and k >= 0 and k < NODE_X )
	{
		ix = Index3D( i, j, k, NODE_X );
		host_node[ix]->active = false;
	}

	return host_node[ix]->active == false;
};

#pragma endregion