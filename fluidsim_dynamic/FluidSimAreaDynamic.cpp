/**
* <Author>      Orlando Chen
* <First>       Jan 07, 2014
* <Last>		Jan 09, 2014
* <File>        FluidSimAreaDynamic.cpp
*/

#include <iostream>
#include <cuda_runtime.h>
#include "FluidSimAreaDynamic.h"
#include "FunctionHelperDynamic.h"

sge::FluidSimProc::FluidSimProc ( fluidsim *fluid )
{
	if ( AllocateResourcePtrs ( fluid ) != SG_RUNTIME_OK )
	{
		FreeResourcePtrs ();
		exit (1);
	}

	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;

	IXi = IXj = IXk = 0;

	ActiveNode ( IXi, IXj, IXk );

	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	ZeroData ();
};

sge::SGRUNTIMEMSG sge::FluidSimProc::AllocateResourcePtrs ( fluidsim *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );

#pragma region allocate memory and buffer on both host and devices

	/* allocate buffer on GPU devices */
	for ( int i = 0; i < DevListNum; i++ )
	{
		/* alarm if cudaMalloc failed */
		static double *ptr;
		if ( cudaMalloc( (void **) &ptr, Sim_Size * sizeof(double) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}
#pragma endregion


#pragma region allocate memory and buffer for volume rendering

	/* allocate host visual with real size */
	size_t size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;
	host_visual = (uchar*) malloc ( sizeof(uchar) * size );

	/* allocate device visual with real size */
	if ( cudaMalloc ((void**)&dev_visual, sizeof(uchar) * size ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}

	/* allocate device small visual with small size */
	size = Sim_Size;
	if ( cudaMalloc ((void**)&dev_smallv, sizeof(uchar) * size ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}

#pragma endregion


#pragma region create discrete node
	int x = fluid->area.uWidth / Grids_X;
	int y = fluid->area.uHeight / Grids_X;
	int z = fluid->area.uDepth / Grids_X;
	if ( x < 0 ) x = 1;
	if ( y < 0 ) y = 1;
	if ( z < 0 ) z = 1;

	size = x * y * z;
	for ( int i = 0; i < size; i++ )
	{
		static node node;
		node.ptrVelU = fluid->area.ptrVelU + i * Sim_Size;
		node.ptrVelV = fluid->area.ptrVelV + i * Sim_Size;
		node.ptrVelW = fluid->area.ptrVelW + i * Sim_Size;
		node.ptrDens = fluid->area.ptrDens + i * Sim_Size;
		node.bActive = false;
		node_list.push_back ( node );
	}

	int index = 0;
	for ( int k = 0; k < z; k++ )
	{
		for ( int j = 0; j < y; j++ )
		{
			for ( int i = 0; i < x; i++ )
			{
				index = i + j * x + k * x * y;

				/* left */
				if ( i >= 1 )
					node_list[index].ptrLeft = &node_list[index-1];
				/* right */
				if ( i <= x - 2 )
					node_list[index].ptrRight = &node_list[index+1];
				/* down */
				if ( j >= 1 )
					node_list[index].ptrDown = &node_list[index-x];
				/* up */
				if ( j <= y - 2 )
					node_list[index].ptrUp = &node_list[index+x];
				/* back */
				if ( k >= 1 )
					node_list[index].ptrBack = &node_list[index-x*y];
				/* front */
				if ( k <= z - 2 )
					node_list[index].ptrFront = &node_list[index+x*y];

				node_list[index].nOffi = i;
				node_list[index].nOffj = j;
				node_list[index].nOffk = k;
#if 0
				printf ( "num: %d | offset: %d %d %d\n", index, node_list[index].nOffi, 
					node_list[index].nOffj, node_list[index].nOffk );
				printf ( "left: %d | right: %d | up: %d | down: %d | front: %d | back: %d \n",
					node_list[index].ptrLeft != NULL,
					node_list[index].ptrRight != NULL,
					node_list[index].ptrUp != NULL,
					node_list[index].ptrDown != NULL,
					node_list[index].ptrFront != NULL,
					node_list[index].ptrBack != NULL );
#endif
			}
		}
	}

#pragma endregion

	// Finally
	return SG_RUNTIME_OK;
}  

void sge::FluidSimProc::FreeResourcePtrs ( void )
{
	for ( int i = 0; i < dev_list.size(); i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	/* release vectors */
	dev_list.empty ( );
	node_list.empty();

	SAFE_FREE_PTR( host_visual );
	cudaFree ( dev_visual );
	cudaFree ( dev_smallv );
}

void sge::FluidSimProc::ZeroData ( void )
{
	for ( int i = 0; i < Sim_Size; i++ )
	{
		host_den [ i ] = 0.f;
		host_u   [ i ] = 0.f;
		host_v   [ i ] = 0.f;
		host_w   [ i ] = 0.f;
	}
	
	if ( cudaMemcpy ( dev_den, host_den,
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
	if ( cudaMemcpy ( dev_u, host_u, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
	if ( cudaMemcpy ( dev_v, host_v, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
	if ( cudaMemcpy ( dev_w, host_w, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
}

void sge::FluidSimProc::CopyDataToDevice ( void )
{
	if ( cudaMemcpy (dev_den, host_den, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
	if ( cudaMemcpy (dev_u, host_u, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
	if ( cudaMemcpy (dev_v, host_v, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
	if ( cudaMemcpy (dev_w, host_w, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void sge::FluidSimProc::CopyDataToHost ( void )
{
	if ( cudaMemcpy ( host_den, dev_den, 
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	if ( cudaMemcpy ( host_u, dev_u, 
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	if ( cudaMemcpy ( host_v, dev_v, 
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
	if ( cudaMemcpy ( host_w, dev_w, 
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void sge::FluidSimProc::ActiveNode ( int i, int j, int k )
{
	if ( i >= 0 and i < Dim ) 
	if ( i >= 0 and i < Dim )
	if ( i >= 0 and i < Dim )
	{
		IXi = i;
		IXj = j;
		IXk = k;
	}

	int index = IXi + IXj * Dim + IXk * Dim * Dim;

	host_den = node_list [ index ].ptrDens;
	host_u   = node_list [ index ].ptrVelU;
	host_v   = node_list [ index ].ptrVelV;
	host_w   = node_list [ index ].ptrVelW;
};