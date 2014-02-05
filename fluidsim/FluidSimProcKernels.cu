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
};