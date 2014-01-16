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

using namespace sge;

FluidSimProc::FluidSimProc( fluidsim *fluid )
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


SGRUNTIMEMSG FluidSimProc::AllocateResourcePtrs( fluidsim *fluid )
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


void FluidSimProc::BuildStructure( void )
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


void FluidSimProc::FreeResourcePtrs( void )
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


void FluidSimProc::ZeroAllBuffer( void )
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


void FluidSimProc::ZeroDevData( void )
{
	extern void hostZeroBuffer( double *grid );

	for ( int i = 0; i < dev_grids.size(); i++ ) hostZeroBuffer( dev_grids[ i ] );
};

#pragma region download & upload data between host and CUDA devices

void FluidSimProc::LeftDataToDevice( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;
	/* upload data to grid */
	if ( host_nodes[ m_index ].ptrLeft isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrLeft;

		/* copy data to grid */
		if ( cudaMemcpy(dev_d_L, ptr->ptrDens, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_u_L, ptr->ptrVelU, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_v_L, ptr->ptrVelV, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_w_L, ptr->ptrVelW, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void FluidSimProc::RightDataToDevice( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;
	/* upload data to grid */
	if ( host_nodes[ m_index ].ptrRight isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrRight;
		
		/* copy data to grid */
		if ( cudaMemcpy(dev_d_R, ptr->ptrDens, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_u_R, ptr->ptrVelU, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_v_R, ptr->ptrVelV, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_w_R, ptr->ptrVelW, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void FluidSimProc::UpDataToDevice( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;
	/* upload data to grid */
	if ( host_nodes[ m_index ].ptrUp isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrUp;

		/* copy data to grid */
		if ( cudaMemcpy(dev_d_U, ptr->ptrDens, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_u_U, ptr->ptrVelU, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_v_U, ptr->ptrVelV, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_w_U, ptr->ptrVelW, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void FluidSimProc::DownDataToDevice( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;
	/* upload data to grid */
	if ( host_nodes[ m_index ].ptrDown isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrDown;

		/* copy data to grid */
		if ( cudaMemcpy(dev_d_D, ptr->ptrDens, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_u_D, ptr->ptrVelU, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_v_D, ptr->ptrVelV, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_w_D, ptr->ptrVelW, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void FluidSimProc::FrontDataToDevice( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;
	/* upload data to grid */
	if ( host_nodes[ m_index ].ptrFront isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrFront;

		/* copy data to grid */
		if ( cudaMemcpy(dev_d_F, ptr->ptrDens, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_u_F, ptr->ptrVelU, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_v_F, ptr->ptrVelV, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_w_F, ptr->ptrVelW, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::BackDataToDevice( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;
	/* upload data to grid */
	if ( host_nodes[ m_index ].ptrBack isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrBack;

		/* copy data to grid */
		if ( cudaMemcpy(dev_d_B, ptr->ptrDens, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_u_B, ptr->ptrVelU, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_v_B, ptr->ptrVelV, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy(dev_w_B, ptr->ptrVelW, size, cudaMemcpyHostToDevice) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::LeftDataToHost( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;

	/* draw data back to host */
	if ( host_nodes[ m_index ].ptrLeft isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrLeft;

		if ( cudaMemcpy( ptr->ptrDens, dev_d_L, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelU, dev_u_L, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelV, dev_v_L, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelW, dev_w_L, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::RightDataToHost( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;

	/* draw data back to host */
	if ( host_nodes[ m_index ].ptrRight isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrRight;

		if ( cudaMemcpy( ptr->ptrDens, dev_d_R, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelU, dev_u_R, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelV, dev_v_R, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelW, dev_w_R, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::UpDataToHost( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;

	/* draw data back to host */
	if ( host_nodes[ m_index ].ptrUp isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrUp;

		if ( cudaMemcpy( ptr->ptrDens, dev_d_U, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelU, dev_u_U, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelV, dev_v_U, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelW, dev_w_U, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::DownDataToHost( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;

	/* draw data back to host */
	if ( host_nodes[ m_index ].ptrDown isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrDown;

		if ( cudaMemcpy( ptr->ptrDens, dev_d_D, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelU, dev_u_D, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelV, dev_v_D, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelW, dev_w_D, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::FrontDataToHost( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;

	/* draw data back to host */
	if ( host_nodes[ m_index ].ptrFront isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrFront;
		
		if ( cudaMemcpy( ptr->ptrDens, dev_d_F, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelU, dev_u_F, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelV, dev_v_F, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelW, dev_w_F, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
};

void FluidSimProc::BackDataToHost( void )
{
	size_t size = sizeof(double) * CUBESIZE_X;

	/* draw data back to host */
	if ( host_nodes[ m_index ].ptrBack isnot NULL )
	{
		node *ptr = host_nodes[ m_index ].ptrBack;

		if ( cudaMemcpy( ptr->ptrDens, dev_d_B, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelU, dev_u_B, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelV, dev_v_B, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
		if ( cudaMemcpy( ptr->ptrVelW, dev_w_B, size, cudaMemcpyDeviceToHost) isnot cudaSuccess )
			goto Error;
	}

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
}

#pragma endregion

void FluidSimProc::CopyDataToDevice( void )
{
	/* upload data to current grid */
	if ( cudaMemcpy (dev_d, host_nodes[ m_index ].ptrDens, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u, host_nodes[ m_index ].ptrVelU, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, host_nodes[ m_index ].ptrVelV, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, host_nodes[ m_index ].ptrVelW, sizeof(double) * CUBESIZE_X,
		cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	/* upload data to neighbouring grids */
	LeftDataToDevice();
	RightDataToDevice();
	UpDataToDevice();
	DownDataToDevice();
	FrontDataToDevice();
	BackDataToDevice();

	goto Success;

Error:
	cudaCheckErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs();
	exit(1);

Success:
	;	
};


void FluidSimProc::CopyDataToHost( void )
{
	/* draw data back */
	if ( cudaMemcpy( host_nodes[ m_index ].ptrDens, dev_d,
		sizeof(double) * CUBESIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy( host_nodes[ m_index ].ptrVelU, dev_u,
		sizeof(double) * CUBESIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy(host_nodes[ m_index ].ptrVelV, dev_v,
		sizeof(double) * CUBESIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy(host_nodes[ m_index ].ptrVelW, dev_w,
		sizeof(double) * CUBESIZE_X, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;

	/* draw data to neighbouring grids */
	LeftDataToHost();
	RightDataToHost();
	UpDataToHost();
	DownDataToHost();
	FrontDataToHost();
	BackDataToHost();

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void FluidSimProc::SelectNode( int i, int j, int k )
{
	host_nodes[ m_index ].bActive = false;
	printf ( "node no.%d is deactive!\n", m_index );

	if ( i >= 0 and i < NODES_X ) 
	if ( j >= 0 and j < NODES_X )
	if ( k >= 0 and k < NODES_X )
	{
		m_index = cudaIndex3D( i, j, k, NODES_X );
		host_nodes[ m_index ].bActive = true;
		printf ("node no.%d is selected!\n", m_index);
	}	
};

void FluidSimProc::SelectNode( int index )
{
	host_nodes[ m_index ].bActive = false;
	printf( "node no.%d is deactive!\n", m_index );

	size_t size = NODES_X * NODES_X * NODES_X;
	if ( index >= 0 and index < size )
	{
		m_index = index;
		host_nodes[ m_index ].bActive = true;
		printf( "node no.%d is selected!\n", m_index );
	}
};