/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 21, 2013
* <Last Time>     Feb 21, 2014
* <File Name>     DataFlooding.cu
*/

#include <iostream>
#include <cuda_runtime.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"

using namespace sge;

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

__global__ void kernelZeroTempBuffers( double *bufs )
{
	GetIndex();

	bufs[threadIdx.x] = 0.f;
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
		helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

	/* upload left buffers */
	if ( node->ptrLeft not_eq nullptr )
	{
		if ( cudaMemcpy( dev_left, container[cudaIndex3D(i-1,j,k,NODES_X)],
			m_node_size, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
		helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit(1);
	}

	/* download left buffers */
	if ( node->ptrLeft not_eq nullptr )
	{
		if ( cudaMemcpy( container[cudaIndex3D(i-1,j,k,NODES_X)], dev_left, 
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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
	
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_left, dev_center, MACRO_LEFT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_right, dev_center, MACRO_RIGHT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_up, dev_center, MACRO_UP );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_down, dev_center, MACRO_DOWN );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_front, dev_center, MACRO_FRONT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_back, dev_center, MACRO_BACK );

	if ( isDensity )
	{
		/* zero temporary buffers */
		kernelZeroTempBuffers <<<1, TPBUFFER_X>>> ( dev_tpbufs );

		/* check */
		kernelSumDensity<<<gridDim, blockDim>>> 
			( dev_tpbufs, dev_center, dev_left, dev_right, dev_up, dev_down, dev_front, dev_back );

		if ( cudaMemcpy( host_tpbufs, dev_tpbufs, sizeof(double) * TPBUFFER_X, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
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