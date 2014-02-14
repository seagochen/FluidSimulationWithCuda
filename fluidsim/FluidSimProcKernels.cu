/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Nov 15, 2013
* <Last Time>     Feb 05, 2014
* <File Name>     FluidSimProc.cu
*/

#include <stdlib.h>
#include <stdio.h>

#define DEVLISTNUM       10
#define dev_u     dev_bufs[0]
#define dev_v     dev_bufs[1]
#define dev_w     dev_bufs[2]
#define dev_div   dev_bufs[3]
#define dev_p     dev_bufs[4]
#define dev_dens  dev_bufs[5]
#define dev_u0    dev_bufs[6]
#define dev_v0    dev_bufs[7]
#define dev_w0    dev_bufs[8]
#define dev_dens0 dev_bufs[9]

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "FluidSimProc.h"

using namespace sge;

FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
{
	/* initialize the parameters of fluid simulation */
	InitParams( fluid );

	/* allocate the space for fluid simulation */
	if ( !AllocateResource() )
	{
		FreeResource();
		printf(" malloc buffers for fluid simulation failed! \n");
		exit(1);
	}

	/* initialize the nodes */
	InitSimNodes();

	/* building structure order */
	BuildOrder();

	/* select and active a node for fluid simulation */
	ActiveNode( 1, 1, 0 );
	
	/* finally, print the state message and zero the data */
	printf( "fluid simulation ready...\n" );
};

void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	/* initilize the status of FPS counter */
	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;
};

__device__ void atomicCopyGridsData( SGSTDGRID *grid0, const SGSTDGRID *grid1, int id )
{
	grid0[id].dens     = grid1[id].dens;
	grid0[id].obstacle = grid1[id].obstacle;
	grid0[id].u        = grid1[id].u;
	grid0[id].v        = grid1[id].v;
	grid0[id].w        = grid1[id].w;
};

__global__ void kernelUploadSTDGrids
	( SGCUDANODES *nodes, const SGSTDGRID *ptrCenter,
	const SGSTDGRID *ptrLeft, const SGSTDGRID *ptrRight,
	const SGSTDGRID *ptrUp,   const SGSTDGRID *ptrDown,
	const SGSTDGRID *ptrFront,const SGSTDGRID *ptrBack )
{
	GetIndex();
	int id = Index(i,j,k);

	atomicCopyGridsData( nodes->ptrCenter, ptrCenter, id );
	atomicCopyGridsData( nodes->ptrLeft, ptrLeft, id );
	atomicCopyGridsData( nodes->ptrRight, ptrRight, id );
	atomicCopyGridsData( nodes->ptrUp, ptrUp, id );
	atomicCopyGridsData( nodes->ptrDown, ptrDown, id );
	atomicCopyGridsData( nodes->ptrFront, ptrFront, id );
	atomicCopyGridsData( nodes->ptrBack, ptrBack, id );
};

__global__ void kernelZeroSTDGrids( SGSTDGRID *grids )
{
	GetIndex();
	int id = Index(i,j,k);

	grids[id].dens = 0.f;
	grids[id].u    = 0.f;
	grids[id].v    = 0.f;
	grids[id].w    = 0.f;
	grids[id].obstacle = SG_BLANK;
};

/* copy host data to CUDA device */
void FluidSimProc::UploadBuffers( void )
{
	cudaDeviceDim3D();
	size_t size = GRIDS_X * GRIDS_X * GRIDS_X;

	if ( host_nodes[m_ix].bActive )
	{
		/* upload buffers to device */
		if ( host_nodes[m_ix].ptrLeft eqt nullptr ) kernelZeroSTDGrids<<<gridDim, blockDim>>>(ptrLeft);
		elif ( cudaMemcpy(ptrLeft, host_nodes[m_ix].ptrLeft->ptrGrids,
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}		

		if ( host_nodes[m_ix].ptrRight eqt nullptr ) kernelZeroSTDGrids<<<gridDim, blockDim>>>(ptrRight);
		elif ( cudaMemcpy(ptrRight, host_nodes[m_ix].ptrRight->ptrGrids,
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		if ( host_nodes[m_ix].ptrFront eqt nullptr ) kernelZeroSTDGrids<<<gridDim, blockDim>>>(ptrFront);
		elif ( cudaMemcpy(ptrFront, host_nodes[m_ix].ptrFront->ptrGrids,
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		if ( host_nodes[m_ix].ptrBack eqt nullptr ) kernelZeroSTDGrids<<<gridDim, blockDim>>>(ptrBack);
		elif ( cudaMemcpy(ptrBack, host_nodes[m_ix].ptrBack->ptrGrids,
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		if ( host_nodes[m_ix].ptrUp eqt nullptr ) kernelZeroSTDGrids<<<gridDim, blockDim>>>(ptrUp);
		elif ( cudaMemcpy(ptrUp, host_nodes[m_ix].ptrUp->ptrGrids,
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		if ( host_nodes[m_ix].ptrDown eqt nullptr ) kernelZeroSTDGrids<<<gridDim, blockDim>>>(ptrDown);
		elif ( cudaMemcpy(ptrDown, host_nodes[m_ix].ptrDown->ptrGrids,
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		if ( cudaMemcpy( ptrCenter, host_nodes[m_ix].ptrGrids, 
			sizeof(SGSTDGRID) * size, cudaMemcpyHostToDevice) != cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		/* Copy data to device nodes for next fluid simulations */
		kernelUploadSTDGrids <<<gridDim, blockDim>>>
			( dev_nodes, ptrCenter, ptrLeft, ptrRight, ptrUp, ptrDown, ptrFront, ptrBack ); 
	}
};

__global__ void kernelDownloadSTDGrids
	( SGSTDGRID *ptrCenter, SGSTDGRID *ptrLeft, SGSTDGRID *ptrRight,
	SGSTDGRID *ptrUp,   SGSTDGRID *ptrDown,
	SGSTDGRID *ptrFront,SGSTDGRID *ptrBack,
	const SGCUDANODES *nodes )
{
	GetIndex();
	int id = Index(i,j,k);

	atomicCopyGridsData( ptrCenter, nodes->ptrCenter, id );
	atomicCopyGridsData( ptrLeft, nodes->ptrLeft, id );
	atomicCopyGridsData( ptrRight, nodes->ptrRight, id );
	atomicCopyGridsData( ptrUp, nodes->ptrUp, id );
	atomicCopyGridsData( ptrDown, nodes->ptrDown, id );
	atomicCopyGridsData( ptrFront, nodes->ptrFront, id );
	atomicCopyGridsData( ptrBack, nodes->ptrBack, id );
};

/* retrieve data back to host */
void FluidSimProc::DownloadBuffers( void )
{
	cudaDeviceDim3D();
	size_t size = GRIDS_X * GRIDS_X * GRIDS_X;

	if ( host_nodes[m_ix].bActive )
	{
		/* retrieve data back */
		kernelDownloadSTDGrids<<<gridDim, blockDim>>>
			( ptrCenter, ptrLeft, ptrRight, ptrUp, ptrDown, ptrFront, ptrBack, dev_nodes );

		if ( cudaMemcpy( host_nodes[m_ix].ptrGrids, ptrCenter,
			sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit(1);
		}

		if ( host_nodes[m_ix].ptrLeft not_eq nullptr )
		{
			if ( cudaMemcpy( host_nodes[m_ix].ptrLeft->ptrGrids, ptrLeft,
				sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
			{
				m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
				FreeResource();
				exit(1);
			}
		}

		if ( host_nodes[m_ix].ptrRight not_eq nullptr )
		{
			if ( cudaMemcpy( host_nodes[m_ix].ptrRight->ptrGrids, ptrRight,
				sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
			{
				m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
				FreeResource();
				exit(1);
			}
		}

		if ( host_nodes[m_ix].ptrUp not_eq nullptr )
		{
			if ( cudaMemcpy( host_nodes[m_ix].ptrUp->ptrGrids, ptrUp,
				sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
			{
				m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
				FreeResource();
				exit(1);
			}
		}

		if ( host_nodes[m_ix].ptrDown not_eq nullptr )
		{
			if ( cudaMemcpy( host_nodes[m_ix].ptrDown->ptrGrids, ptrDown,
				sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
			{
				m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
				FreeResource();
				exit(1);
			}
		}

		if ( host_nodes[m_ix].ptrFront not_eq nullptr )
		{
			if ( cudaMemcpy( host_nodes[m_ix].ptrFront->ptrGrids, ptrFront,
				sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
			{
				m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
				FreeResource();
				exit(1);
			}
		}

		if ( host_nodes[m_ix].ptrBack not_eq nullptr )
		{
			if ( cudaMemcpy( host_nodes[m_ix].ptrBack->ptrGrids, ptrBack,
				sizeof(SGSTDGRID) * size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
			{
				m_helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
				FreeResource();
				exit(1);
			}
		}
	};
};

void FluidSimProc::DeactiveNode( int i, int j, int k )
{
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		m_ix = cudaIndex3D( i, j, k, NODES_X );

		if ( host_nodes[m_ix].bActive )
		{
			host_nodes[m_ix].bActive = false;
			printf ( "node no.%d is deactived!\n", m_ix );
		}
	}
};

void FluidSimProc::ActiveNode( int i, int j, int k )
{
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		m_ix = cudaIndex3D( i, j, k, NODES_X );

		if ( !host_nodes[m_ix].bActive )
		{
			host_nodes[m_ix].bActive = true;
			printf ( "node no.%d is actived!\n", m_ix );
		}
	}	
};

void FluidSimProc::BuildOrder( void )
{
	printf( "structure:\n" );
	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				int index = cudaIndex3D( i, j, k, NODES_X );

				if ( index >= host_nodes.size() or index < 0 )
				{
					printf ( "index out of range! %s, line: %d \n", __FILE__, __LINE__ );
					exit ( 1 );
				}

				/* left */
				if ( i >= 1 )
					host_nodes[index].ptrLeft = &host_nodes[index-1];
				/* right */
				if ( i <= NODES_X - 2 )
					host_nodes[index].ptrRight = &host_nodes[index+1];
				/* down */
				if ( j >= 1 )
					host_nodes[index].ptrDown = &host_nodes[index-NODES_X];
				/* up */
				if ( j <= NODES_X - 2 )
					host_nodes[index].ptrUp = &host_nodes[index+NODES_X];
				/* back */
				if ( k >= 1 )
					host_nodes[index].ptrBack = &host_nodes[index-NODES_X*NODES_X];
				/* front */
				if ( k <= NODES_X - 2 )
					host_nodes[index].ptrFront = &host_nodes[index+NODES_X*NODES_X];

				host_nodes[index].n3Pos.x = i;
				host_nodes[index].n3Pos.y = j;
				host_nodes[index].n3Pos.z = k;

				printf ( "no: %d | offset: %d%d%d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index,
					host_nodes[index].n3Pos.x, 
					host_nodes[index].n3Pos.y, 
					host_nodes[index].n3Pos.z,
					host_nodes[index].ptrLeft != NULL,
					host_nodes[index].ptrRight != NULL,
					host_nodes[index].ptrUp != NULL,
					host_nodes[index].ptrDown != NULL,
					host_nodes[index].ptrFront != NULL,
					host_nodes[index].ptrBack != NULL );
			}
		}
	}

	printf( "-----------------------------------------------\n" );
};


void FluidSimProc::InitSimNodes( void )
{
	/* Get node index */
	int nID = cudaIndex3D( 1, 1, 0, NODES_X );

	const int half = GRIDS_X / 2;
	
	for ( int i = 0; i < GRIDS_X; i++ )
	{
		for ( int j = 0; j < GRIDS_X; j++ )
		{
			for ( int k = 0; k < GRIDS_X; k++ )
			{
				host_nodes[ nID ].ptrGrids[ Index(i, j, k) ].obstacle = SG_BD BD_BLANK;

				if ( j < 1 )
					if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
						host_nodes[ m_index ].ptrObs[ Index(i, j, k) ] = BD_SOURCE;
			}
		}
	}
};