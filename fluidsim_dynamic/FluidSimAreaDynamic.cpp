/**
* <Author>      Orlando Chen
* <First>       Nov 15, 2013
* <Last>		Jan 12, 2014
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

	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	SelectNode (0, 0, 0);
	ZeroData ();
};

sge::SGRUNTIMEMSG sge::FluidSimProc::AllocateResourcePtrs ( fluidsim *fluid )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );

#pragma region allocate memory on both host, device and volume

	// Allocate memory on host
	size_t size = Node_X * Node_X * Node_X;
	for ( int i = 0; i < size; i++ )
	{
		static node node;
		node.ptrDens = (double*) malloc ( Sim_Size * sizeof(double) );
		node.ptrVelU = (double*) malloc ( Sim_Size * sizeof(double) );
		node.ptrVelV = (double*) malloc ( Sim_Size * sizeof(double) );
		node.ptrVelW = (double*) malloc ( Sim_Size * sizeof(double) );
		node.ptrP    = (double*) malloc ( Sim_Size * sizeof(double) );
		node.ptrDiv  = (double*) malloc ( Sim_Size * sizeof(double) );
		node_list.push_back ( node );
	}

	// Allocate memory on GPU devices
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static double *ptr;
		if ( cudaMalloc( (void **) &ptr, Sim_Size * sizeof(double) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;	
	host_visual = (uchar*) malloc ( sizeof(uchar) * size );
	if ( cudaMalloc ((void**)&dev_visual, sizeof(uchar) * size ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}
	
#pragma endregion


#pragma region assign node position

	for ( int i = 0; i < Node_X; i++ )
	{
		for ( int j = 0; j < Node_X; j++ )
		{
			for ( int k = 0; k < Node_X; k++ )
			{
				int index = i + j * Node_X + k * Node_X * Node_X;

				/* left */
				if ( i >= 1 )
					node_list[index].ptrLeft = &node_list[index-1];
				/* right */
				if ( i <= Node_X - 2 )
					node_list[index].ptrRight = &node_list[index+1];
				/* down */
				if ( j >= 1 )
					node_list[index].ptrDown = &node_list[index-Node_X];
				/* up */
				if ( j <= Node_X - 2 )
					node_list[index].ptrUp = &node_list[index+Node_X];
				/* back */
				if ( k >= 1 )
					node_list[index].ptrBack = &node_list[index-Node_X*Node_X];
				/* front */
				if ( k <= Node_X - 2 )
					node_list[index].ptrFront = &node_list[index+Node_X*Node_X];

				node_list[index].i = i;
				node_list[index].j = j;
				node_list[index].k = k;

				printf ( "no: %d | offset: %d%d%d "
					"| L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index, node_list[index].i, 
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

	// Finally
	return SG_RUNTIME_OK;
}  

void sge::FluidSimProc::FreeResourcePtrs ( void )
{
	size_t size = Node_X * Node_X * Node_X;

	for ( int i = 0; i < size; i++ )
	{
		SAFE_FREE_PTR ( node_list[ i ].ptrDens );
		SAFE_FREE_PTR ( node_list[ i ].ptrVelU );
		SAFE_FREE_PTR ( node_list[ i ].ptrVelV );
		SAFE_FREE_PTR ( node_list[ i ].ptrVelW );
		SAFE_FREE_PTR ( node_list[ i ].ptrP );
		SAFE_FREE_PTR ( node_list[ i ].ptrDiv );
	}
	node_list.empty ( );

	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );

	SAFE_FREE_PTR( host_visual );
	cudaFree ( dev_visual );
}

void sge::FluidSimProc::ZeroData ( void )
{
	size_t size = Node_X * Node_X * Node_X;
	for ( int ii = 0; ii < size; ii++ )
	for ( int i = 0; i < Sim_Size; i++ )
	{
		node_list[ ii ].ptrDens[ i ] = 0.f;
		node_list[ ii ].ptrVelU[ i ] = 0.f;
		node_list[ ii ].ptrVelV[ i ] = 0.f;
		node_list[ ii ].ptrVelW[ i ] = 0.f;
		node_list[ ii ].ptrP[ i ]    = 0.f;
		node_list[ ii ].ptrDiv[ i ]  = 0.f;
	}

	if ( cudaMemcpy (dev_den, node_list[ IX ].ptrDens, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u, node_list[ IX ].ptrVelU, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, node_list[ IX ].ptrVelV, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, node_list[ IX ].ptrVelW, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_p, node_list[ IX ].ptrP, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_div, node_list[ IX ].ptrDiv, 
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
	if ( cudaMemcpy (dev_den, node_list[ IX ].ptrDens, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u, node_list[ IX ].ptrVelU, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, node_list[ IX ].ptrVelV, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, node_list[ IX ].ptrVelW, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_p, node_list[ IX ].ptrP, 
		sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_div, node_list[ IX ].ptrDiv, 
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
	if ( cudaMemcpy (node_list[ IX ].ptrDens, dev_den,
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrVelU, dev_u,
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrVelV, dev_v,
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrVelW, dev_w,
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrP, dev_p,
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (node_list[ IX ].ptrDiv, dev_div,
		sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void sge::FluidSimProc::SelectNode ( int i, int j, int k )
{
	if ( i >= 0 and i < Node_X ) 
	if ( i >= 0 and i < Node_X )
	if ( i >= 0 and i < Node_X )
	{
		offi = i;
		offj = j;
		offk = k;
		IX = offi + offj * Node_X + offk * Node_X * Node_X;
	}	
};

void sge::FluidSimProc::SelectNode ( int index )
{
	size_t size = Node_X * Node_X * Node_X;
	if ( index >= 0 and index < size )
		IX = index;
};