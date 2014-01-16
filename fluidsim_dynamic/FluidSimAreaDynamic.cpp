/**
* <Author>      Orlando Chen
* <First>       Nov 15, 2013
* <Last>		Jan 16, 2014
* <File>        FluidSimAreaDynamic.cpp
*/

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "FluidSimAreaDynamic.h"
#include "FunctionHelperDynamic.h"

sge::FluidSimProc::FluidSimProc( fluidsim *fluid )
{
	/* allocate the space for fluid simulation */
	if ( AllocateResourcePtrs( fluid ) != SG_RUNTIME_OK )
	{
		FreeResourcePtrs();
		exit(1);
	}

	/* building data structure */
	BuildStructure();

	/* initilize the status of FPS counter */
	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;

	/* select a node as initilize stage */
	SelectNode( 10 );

	/* finally, print the state message and zero the data */
	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	ZeroAllBuffer();
};


sge::SGRUNTIMEMSG sge::FluidSimProc::AllocateResourcePtrs( fluidsim *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );


	/* allocate memory on host */
	size_t size = NODES_X * NODES_X * NODES_X;
	for ( int i = 0; i < size; i++ )
	{
		static node node;
		node.ptrDens = (double*) calloc( CUBESIZE_X, sizeof(double) );
		node.ptrVelU = (double*) calloc( CUBESIZE_X, sizeof(double) );
		node.ptrVelV = (double*) calloc( CUBESIZE_X, sizeof(double) );
		node.ptrVelW = (double*) calloc( CUBESIZE_X, sizeof(double) );
		if ( node.ptrDens is NULL or node.ptrVelU is NULL 
			or node.ptrVelV is NULL or node.ptrVelW is NULL )
		{
			printf( "calloc memory failed, file: %f, line: %d\n", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
		/* push node to vector */
		host_nodes.push_back( node );
	}


	/* allocate memory on GPU devices */
	for ( int i = 0; i < DevGridsNum; i++ )
	{
		/* alarm if cudaMalloc failed */
		static double *ptr;
		if ( cudaMalloc( (void **)&ptr, CUBESIZE_X * sizeof(double) ) isnot cudaSuccess )
		{
			cudaCheckErrors( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
		/* push grids to vector */
		dev_grids.push_back( ptr );
	}


	/* allocate the buffer for temporary faces storing */
	for ( int i = 0; i < DevFacesNum; i++ )
	{
		/* alarm if cudaMalloc failed */
		static double *ptr;
		if ( cudaMalloc( (void**)&ptr, FACESIZE_X * sizeof(double) ) isnot cudaSuccess )
		{
			cudaCheckErrors( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
		/* push faces to vector */
		dev_faces.push_back( ptr );
	}


	/* allocate temporary storing buffer on both host and device */
	host_ibuf = (int*) calloc( TPBUFFER_X, sizeof(int) );
	host_fbuf = (double*) calloc( TPBUFFER_X, sizeof(double) );
	/* alarm if calloc is failed */
	if ( host_ibuf is NULL )
	{
		printf( "calloc host_ibuf failed!\n" );
		return SG_MALLOC_SPACE_FAILED;
	}
	if ( host_fbuf is NULL )
	{
		printf( "calloc host_fbuf failed!\n" );
		return SG_MALLOC_SPACE_FAILED;
	}
	/* alarm if cudaMalloc is failed */
	if ( cudaMalloc( (void**)&dev_ibuf, sizeof(int) * TPBUFFER_X ) isnot cudaSuccess )
	{
		cudaCheckErrors( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_MALLOC_SPACE_FAILED;
	}
	if ( cudaMalloc( (void**)&dev_fbuf, sizeof(double) * TPBUFFER_X) isnot cudaSuccess )
	{
		cudaCheckErrors( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_MALLOC_SPACE_FAILED;
	}


	/* allocate volume buffer */
	size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;
	
	host_visual = (uchar*) calloc( size, sizeof(uchar) );
	/* alarm if calloc failed */
	if ( host_visual is NULL )
	{
		printf( "calloc host_visual failed!\n" );
		return SG_MALLOC_SPACE_FAILED;
	}
	/* alarm if cudaMalloc failed */
	if ( cudaMalloc ((void**)&dev_visual, sizeof(uchar) * size ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_MALLOC_SPACE_FAILED;
	}


	/* finally */
	return SG_RUNTIME_OK;
}  


void sge::FluidSimProc::BuildStructure( void )
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

				host_nodes[index].i = i;
				host_nodes[index].j = j;
				host_nodes[index].k = k;

				printf ( "no: %d | offset: %d%d%d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index,
					host_nodes[index].i, 
					host_nodes[index].j, 
					host_nodes[index].k,
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


void sge::FluidSimProc::FreeResourcePtrs( void )
{
	/* release nodes list */
	size_t size = NODES_X * NODES_X * NODES_X;
	for ( int i = 0; i < size; i++ )
	{
		SAFE_FREE_PTR( host_nodes[ i ].ptrDens );
		SAFE_FREE_PTR( host_nodes[ i ].ptrVelU );
		SAFE_FREE_PTR( host_nodes[ i ].ptrVelV );
		SAFE_FREE_PTR( host_nodes[ i ].ptrVelW );
	}
	host_nodes.empty();
	printf( "nodes released!\n" );


	/* release device grids */
	for ( int i = 0; i < DevGridsNum; i++ ) cudaFree( dev_grids[ i ] );
	dev_grids.empty ();
	printf( "device grids released!\n" );


	/* release faces */
	for ( int i = 0; i < DevFacesNum; i++ ) cudaFree( dev_faces[ i ] );
	printf( "device faces released!\n" );

	
	/* release temproary buffers */
	SAFE_FREE_PTR( host_ibuf );
	SAFE_FREE_PTR( host_fbuf );
	cudaFree( dev_ibuf );
	cudaFree( dev_fbuf );
	printf( "temporary buffer released!\n" );


	/* release visual buffers */
	SAFE_FREE_PTR ( host_visual );
	cudaFree ( dev_visual );
	printf( "3D visual buffer released!\n" );
};


void sge::FluidSimProc::ZeroAllBuffer( void )
{
	extern void hostZeroBuffer( double *grid );
	extern void hostZeroBuffer( unsigned char *grid, int const offi, int const offj, int const offk );

	/* zero device grids one by one */
	for ( int i = 0; i < dev_grids.size(); i++ ) hostZeroBuffer( dev_grids [ i ] );

	/* zero each node one by one */
	for ( int i = 0; i < host_nodes.size(); i++ )
	{
		/* draw blank data back */
		if ( cudaMemcpy( host_nodes[ i ].ptrDens, dev_t, sizeof(double) * CUBESIZE_X,
			cudaMemcpyDeviceToHost) != cudaSuccess ) goto Error;
		if ( cudaMemcpy (host_nodes[ i ].ptrVelU, dev_t, sizeof(double) * CUBESIZE_X,
			cudaMemcpyDeviceToHost) != cudaSuccess ) goto Error;
		if ( cudaMemcpy (host_nodes[ i ].ptrVelV, dev_t, sizeof(double) * CUBESIZE_X,
			cudaMemcpyDeviceToHost) != cudaSuccess ) goto Error;
		if ( cudaMemcpy (host_nodes[ i ].ptrVelW, dev_t, sizeof(double) * CUBESIZE_X,
			cudaMemcpyDeviceToHost) != cudaSuccess ) goto Error;
		
		/* and zero visual buffer */
		int offseti = host_nodes[ i ].i * GRIDS_X;
		int offsetj = host_nodes[ i ].j * GRIDS_X;
		int offsetk = host_nodes[ i ].k * GRIDS_X;
		hostZeroBuffer( dev_visual, offseti, offsetj, offsetk );
	}

	size_t size = VOLUME_X * VOLUME_X * VOLUME_X;
	if ( cudaMemcpy (host_visual, dev_visual, sizeof(uchar) * size, 
		cudaMemcpyDeviceToHost ) != cudaSuccess ) goto Error;

	goto Success;

Error:
	cudaCheckErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs();
	exit(1);

Success:
	;
}


void sge::FluidSimProc::ZeroDevData( void )
{
	extern void hostZeroBuffer( double *grid );

	for ( int i = 0; i < dev_grids.size(); i++ ) hostZeroBuffer( dev_grids[ i ] );
};


void sge::FluidSimProc::CopyDataToDevice( void )
{
	/* draw data to current grid */
	if ( cudaMemcpy (dev_den, host_nodes[ index ].ptrDens, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess ) goto Error;
	if ( cudaMemcpy (dev_u, host_nodes[ index ].ptrVelU, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess ) goto Error;
	if ( cudaMemcpy (dev_v, host_nodes[ index ].ptrVelV, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess ) goto Error;
	if ( cudaMemcpy (dev_w, host_nodes[ index ].ptrVelW, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess ) goto Error;

	/* draw data to neighbouring cells */
	if ( host_nodes[ index ].ptrLeft isnot NULL ) // left
		if ( cudaMemcpy (dev_L, host_nodes[ index ].ptrLeft->ptrDens,
			sizeof(double) * CUBESIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
			goto Error;

	if ( host_nodes[ index ].ptrRight isnot NULL ) // right
		if ( cudaMemcpy (dev_R, host_nodes[ index ].ptrRight->ptrDens,
			sizeof(double) * CUBESIZE_X, cudaMemcpyHostToDevice) != cudaSuccess ) 
			goto Error;

	if ( host_nodes[ index ].ptrUp isnot NULL ) // up
		if ( cudaMemcpy (dev_U, host_nodes[ index ].ptrUp->ptrDens,
			sizeof(double) * CUBESIZE_X, cudaMemcpyHostToDevice) != cudaSuccess ) 
			goto Error;

	if ( host_nodes[ index ].ptrDown isnot NULL ) // down
		if ( cudaMemcpy (dev_D, host_nodes[ index ].ptrDown->ptrDens,
			sizeof(double) * CUBESIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
			goto Error;

	if ( host_nodes[ index ].ptrFront isnot NULL ) // front
		if ( cudaMemcpy (dev_F, host_nodes[ index ].ptrFront->ptrDens,
			sizeof(double) * CUBESIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
			goto Error;

	if ( host_nodes[ index ].ptrBack isnot NULL ) // back
		if ( cudaMemcpy (dev_B, host_nodes[ index ].ptrBack->ptrDens,
			sizeof(double) * CUBESIZE_X, cudaMemcpyHostToDevice) != cudaSuccess )
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
	if ( cudaMemcpy (node_list[ index ].ptrDens, dev_den,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ index ].ptrVelU, dev_u,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ index ].ptrVelV, dev_v,
		sizeof(double) * SIMSIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ index ].ptrVelW, dev_w,
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