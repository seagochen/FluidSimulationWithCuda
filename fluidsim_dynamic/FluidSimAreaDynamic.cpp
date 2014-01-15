/**
* <Author>      Orlando Chen
* <First>       Nov 15, 2013
* <Last>		Jan 15, 2014
* <File>        FluidSimAreaDynamic.cpp
*/

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "FluidSimAreaDynamic.h"
#include "FunctionHelperDynamic.h"

sge::FluidSimProc::FluidSimProc( fluidsim *fluid )
{
	if ( AllocateResourcePtrs( fluid ) != SG_RUNTIME_OK )
	{
		FreeResourcePtrs();
		exit(1);
	}

	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;

	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	SelectNode( 10 );
	ZeroAllBuffer();

	// finally, zero the temporary variable buffer
	for ( int i = 0; i < TPBUFFER_X; i++ )
		host_buf[ i ] = 0.f;
	if ( cudaMemcpy(dev_buf, host_buf, 
		sizeof(double) * TPBUFFER_X, cudaMemcpyHostToDevice) != cudaSuccess )
	{
		cudaCheckErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		exit( 1 );
	}
};

sge::SGRUNTIMEMSG sge::FluidSimProc::AllocateResourcePtrs( fluidsim *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );

#pragma region allocate memory on both host, device and volume

	/* allocate memory on host */
	size_t size = NODES_X * NODES_X * NODES_X;
	for ( int i = 0; i < size; i++ )
	{
		static node node;
		node.ptrDens = (double*) malloc ( SIMSIZE_X * sizeof(double) );
		node.ptrVelU = (double*) malloc ( SIMSIZE_X * sizeof(double) );
		node.ptrVelV = (double*) malloc ( SIMSIZE_X * sizeof(double) );
		node.ptrVelW = (double*) malloc ( SIMSIZE_X * sizeof(double) );
		node_list.push_back ( node );
	}

	/* allocate memory on GPU devices */
	for ( int i = 0; i < DevListNum; i++ )
	{
		/* alarm if cudaMalloc failed */
		static double *ptr;
		if ( cudaMalloc( (void **) &ptr, SIMSIZE_X * sizeof(double) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	/* allocate visual buffer */
	size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;	
	host_visual = (uchar*) malloc ( sizeof(uchar) * size );
	if ( cudaMalloc ((void**)&dev_visual, sizeof(uchar) * size ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}
	
#pragma endregion

#pragma region assign node position

	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				int index = i + j * NODES_X + k * NODES_X * NODES_X;

				if ( index >= node_list.size() or index < 0 )
				{
					printf ( "index out of range! %s, line: %d \n", __FILE__, __LINE__ );
					exit ( 1 );
				}

				/* left */
				if ( i >= 1 )
					node_list[index].ptrLeft = &node_list[index-1];
				/* right */
				if ( i <= NODES_X - 2 )
					node_list[index].ptrRight = &node_list[index+1];
				/* down */
				if ( j >= 1 )
					node_list[index].ptrDown = &node_list[index-NODES_X];
				/* up */
				if ( j <= NODES_X - 2 )
					node_list[index].ptrUp = &node_list[index+NODES_X];
				/* back */
				if ( k >= 1 )
					node_list[index].ptrBack = &node_list[index-NODES_X*NODES_X];
				/* front */
				if ( k <= NODES_X - 2 )
					node_list[index].ptrFront = &node_list[index+NODES_X*NODES_X];

				node_list[index].i = i;
				node_list[index].j = j;
				node_list[index].k = k;

				printf ( "no: %d | offset: %d%d%d "
					"| L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index,
					node_list[index].i, 
					node_list[index].j, 
					node_list[index].k,
					node_list[index].ptrLeft != NULL,
					node_list[index].ptrRight != NULL,
					node_list[index].ptrUp != NULL,
					node_list[index].ptrDown != NULL,
					node_list[index].ptrFront != NULL,
					node_list[index].ptrBack != NULL );
			}
		}
	}

#pragma endregion

#pragma region allocate space for temporary buffers
	host_buf = (double *) malloc ( sizeof(double) * TPBUFFER_X );
	if ( cudaMalloc ( (void**)&dev_buf, sizeof(double) * TPBUFFER_X ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}
#pragma endregion

	/* finally */
	return SG_RUNTIME_OK;
}  

void sge::FluidSimProc::FreeResourcePtrs( void )
{
	size_t size = NODES_X * NODES_X * NODES_X;

	for ( int i = 0; i < size; i++ )
	{
		SAFE_FREE_PTR ( node_list[ i ].ptrDens );
		SAFE_FREE_PTR ( node_list[ i ].ptrVelU );
		SAFE_FREE_PTR ( node_list[ i ].ptrVelV );
		SAFE_FREE_PTR ( node_list[ i ].ptrVelW );
	}
	node_list.empty ();

	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ();

	SAFE_FREE_PTR ( host_visual );
	SAFE_FREE_PTR ( host_buf );
	cudaFree ( dev_visual );
	cudaFree ( dev_buf );
}

void sge::FluidSimProc::ZeroAllBuffer( void )
{
	extern void hostZeroBuffer( double *grid );
	extern void hostZeroBuffer( unsigned char *grid, int const offi, int const offj, int const offk );

	for ( int i = 0; i < dev_list.size(); i++ )
		hostZeroBuffer( dev_list [ i ] );

	/* zero each node one by one */
	for ( int i = 0; i < node_list.size(); i++ )
	{
		if ( cudaMemcpy (node_list[ i ].ptrDens, dev_0,
			sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
			goto Error;
		if ( cudaMemcpy (node_list[ i ].ptrVelU, dev_0,
			sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
			goto Error;
		if ( cudaMemcpy (node_list[ i ].ptrVelV, dev_0,
			sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
			goto Error;
		if ( cudaMemcpy (node_list[ i ].ptrVelW, dev_0,
			sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
			goto Error;
		
		/* and zero visual buffer */
		int offseti = node_list[ i ].i * GRIDS_X;
		int offsetj = node_list[ i ].j * GRIDS_X;
		int offsetk = node_list[ i ].k * GRIDS_X;
		hostZeroBuffer( dev_visual, offseti, offsetj, offsetk );
	}

	size_t size = VOLUME_X * VOLUME_X * VOLUME_X;
	if ( cudaMemcpy (host_visual, dev_visual, sizeof(uchar) * size, 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
}

void sge::FluidSimProc::ZeroDevData( void )
{
	extern void hostZeroBuffer( double *grid );

	for ( int i = 0; i < dev_list.size(); i++ )
		hostZeroBuffer( dev_list [ i ] );
};

void sge::FluidSimProc::CopyDataToDevice( void )
{
	if ( cudaMemcpy (dev_den, node_list[ IX ].ptrDens, 
		sizeof(double) * SIMSIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u, node_list[ IX ].ptrVelU, 
		sizeof(double) * SIMSIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, node_list[ IX ].ptrVelV, 
		sizeof(double) * SIMSIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, node_list[ IX ].ptrVelW, 
		sizeof(double) * SIMSIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void sge::FluidSimProc::CopyDataToHost( void )
{
	if ( cudaMemcpy (node_list[ IX ].ptrDens, dev_den,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrVelU, dev_u,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrVelV, dev_v,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrVelW, dev_w,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void sge::FluidSimProc::SelectNode( int i, int j, int k )
{
	node_list [ IX ].bActive = false;
	printf ("node no.%d is deactive!\n", IX);

	if ( i >= 0 and i < NODES_X ) 
	if ( j >= 0 and j < NODES_X )
	if ( k >= 0 and k < NODES_X )
	{
		IX = i + j * NODES_X + k * NODES_X * NODES_X;
		node_list [ IX ].bActive = true;
		printf ("node no.%d is selected!\n", IX);
	}	
};

void sge::FluidSimProc::SelectNode( int index )
{
	node_list [ IX ].bActive = false;
	printf ("node no.%d is deactive!\n", IX);

	size_t size = NODES_X * NODES_X * NODES_X;
	if ( index >= 0 and index < size )
	{
		IX = index;
		node_list [ IX ].bActive = true;
		printf ("node no.%d is selected!\n", IX);
	}
};