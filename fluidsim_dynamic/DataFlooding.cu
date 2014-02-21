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

#define MACRO_DENSITY     0
#define MACRO_VELOCITY_U  1
#define MACRO_VELOCITY_V  2
#define MACRO_VELOCITY_W  3
#define MACRO_SIMPLE      4

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_SOURCE     1
#define MACRO_BOUNDARY_OBSTACLE 100

#define MACRO_CENTER     0
#define MACRO_LEFT       1
#define MACRO_RIGHT      2
#define MACRO_UP         3
#define MACRO_DOWN       4
#define MACRO_FRONT      5
#define MACRO_BACK       6

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

__global__ void kernelSumDensity( double *grids, double *buffs, const int ops )
{
	GetIndex();

#define num_dens_center  buffs[ 0 ]
#define num_dens_left    buffs[ 1 ]
#define num_dens_right   buffs[ 2 ]
#define num_dens_up      buffs[ 3 ]
#define num_dens_down    buffs[ 4 ]
#define num_dens_front   buffs[ 5 ]
#define num_dens_back    buffs[ 6 ]
 	
	if ( grids[Index(i,j,k)] > 0.f )
 	{
		switch ( ops )
		{
		case MACRO_LEFT:
			num_dens_left += grids[Index(i,j,k)];
			break;
		case MACRO_RIGHT:
			num_dens_right += grids[Index(i,j,k)];
			break;
		case MACRO_UP:
			num_dens_up += grids[Index(i,j,k)];
			break;
		case MACRO_DOWN:
			num_dens_down += grids[Index(i,j,k)];
			break;
		case MACRO_FRONT:
			num_dens_front += grids[Index(i,j,k)];
			break;
		case MACRO_BACK:
			num_dens_back += grids[Index(i,j,k)];
			break;
		case MACRO_CENTER:
			num_dens_center += grids[Index(i,j,k)];
			break;

		default:
			break;
		}
 	}
#undef num_dens_center
#undef num_dens_left  
#undef num_dens_right 
#undef num_dens_up    
#undef num_dens_down  
#undef num_dens_front 
#undef num_dens_back 

};

__global__ void kernelZeroBuffers
	( double *center, double *left, double *right, double *front, double *back, double *up, double *down )
{
	GetIndex();

	int ix = Index(i,j,k);
	center[ix] = left[ix] = right[ix] = up[ix] = down[ix] = front[ix] = back[ix] = 0.f;
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
//	for ( int i = 0; i < NODES_X; i++ )
//	{
//		for ( int j = 0; j < NODES_X; j++ )
//		{
//			for ( int k = 0; k < NODES_X; k++ )
//			{
//				if ( SelectNode( i, j, k ) )
//				{
//					DataFlooding( host_density, true );
//					DataFlooding( host_velocity_u, false );
//					DataFlooding( host_velocity_v, false );
//					DataFlooding( host_velocity_w, false );
//				}
//			}
//		}
//	}
};

void FluidSimProc::DataFlooding( vector<double*> container, bool bDens )
{
	cudaDeviceDim3D();
	
	kernelZeroBuffers cudaDevice(gridDim, blockDim)
		( dev_center, dev_left, dev_right, dev_up, dev_down, dev_front, dev_back );
 
 	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );
	cudaMemcpy( dev_center, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	
 	if ( nPos.x - 1 >= 0 ) // left
 	{
		ix = cudaIndex3D( nPos.x-1, nPos.y, nPos.z, NODES_X );
		cudaMemcpy( dev_left, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.x + 1 < NODES_X ) // right
 	{
 		ix = cudaIndex3D( nPos.x + 1, nPos.y, nPos.z, NODES_X );
		cudaMemcpy( dev_right, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.y - 1 >= 0 ) // down
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y - 1, nPos.z, NODES_X );
		cudaMemcpy( dev_down, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.y + 1 < NODES_X ) // up
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y + 1, nPos.z, NODES_X );
		cudaMemcpy( dev_up, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.z - 1 >= 0 ) // back
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z - 1, NODES_X );
		cudaMemcpy( dev_back, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.z + 1 < NODES_X ) // front
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z + 1, NODES_X );
		cudaMemcpy( dev_front, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}

	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_left, dev_center, MACRO_LEFT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_right, dev_center, MACRO_RIGHT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_up, dev_center, MACRO_UP );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_down, dev_center, MACRO_DOWN );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_front, dev_center, MACRO_FRONT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_back, dev_center, MACRO_BACK );

	if ( bDens )
 	{
		for ( int i = 0; i < TPBUFFER_X; i++)
		{
			host_tpbufs[i] = 0.f;
		}
		if ( cudaMemcpy( dev_tpbufs, host_tpbufs, 
			sizeof(double) * TPBUFFER_X, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}

		kernelSumDensity<<<gridDim, blockDim>>> ( dev_center, dev_tpbufs, MACRO_CENTER );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_left, dev_tpbufs, MACRO_LEFT );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_right, dev_tpbufs, MACRO_RIGHT );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_up, dev_tpbufs, MACRO_UP );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_down, dev_tpbufs, MACRO_DOWN );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_front, dev_tpbufs, MACRO_FRONT );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_back, dev_tpbufs, MACRO_BACK );
		
		if ( cudaMemcpy( host_tpbufs, dev_tpbufs, sizeof(double) * TPBUFFER_X, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}
		
#define num_dens_center  host_tpbufs[ 0 ]
#define num_dens_left    host_tpbufs[ 1 ]
#define num_dens_right   host_tpbufs[ 2 ]
#define num_dens_up      host_tpbufs[ 3 ]
#define num_dens_down    host_tpbufs[ 4 ]
#define num_dens_front   host_tpbufs[ 5 ]
#define num_dens_back    host_tpbufs[ 6 ]

#if 0
		system( "cls" );
		printf( "center:  %f\n", num_dens_center );
		printf( "left:    %f\n", num_dens_left );
		printf( "right:   %f\n", num_dens_right );
		printf( "up:      %f\n", num_dens_up );
		printf( "down:    %f\n", num_dens_down );
		printf( "front:   %f\n", num_dens_front );
		printf( "back:    %f\n", num_dens_back );
#endif
 	}

	/* retrieve data back to node if density filled */
	if ( nPos.x - 1 >= 0 and num_dens_left > 0.f ) // left
 	{
		ix = cudaIndex3D( nPos.x-1, nPos.y, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_left, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.x + 1 < NODES_X and num_dens_right > 0.f ) // right
 	{
 		ix = cudaIndex3D( nPos.x + 1, nPos.y, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_right, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.y - 1 >= 0 and num_dens_down > 0.f ) // down
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y - 1, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_down, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.y + 1 < NODES_X and num_dens_up > 0.f ) // up
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y + 1, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_up, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.z - 1 >= 0 and num_dens_back > 0.f ) // back
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z - 1, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_back, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.z + 1 < NODES_X and num_dens_front > 0.f ) // front
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z + 1, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_front, m_node_size, cudaMemcpyDeviceToHost );
 	}

#undef num_dens_center
#undef num_dens_left  
#undef num_dens_right 
#undef num_dens_up    
#undef num_dens_down  
#undef num_dens_front 
#undef num_dens_back
};