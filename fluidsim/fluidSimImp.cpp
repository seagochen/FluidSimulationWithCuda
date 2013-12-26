#include <iostream>

#include <cuda_runtime.h>
#include "fluidsim.h"

using namespace sge;

FluidSimProc::FluidSimProc ( fluidsim *fluid )
{
	if ( AllocateResourcePtrs ( fluid ) != SG_RUNTIME_OK )
	{
		FreeResourcePtrs ();
		exit (1);
	}

	times = 0;

	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	ZeroData ();
};

SGRUNTIMEMSG FluidSimProc::AllocateResourcePtrs ( fluidsim *fluid )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices" );

	// Allocate memory on host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static int *ptr;
		ptr = (int*) malloc ( Simul_Size * sizeof(float) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
			return SG_RUNTIME_FALSE;
		}
	}

	// Allocate memory on GPU devices
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static int *ptr;
		if ( cudaMalloc( (void **) &ptr, Simul_Size * sizeof(int) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}
	
	host_data = (GLubyte*) malloc (sizeof(GLubyte) * (fluid->nVolDepth * fluid->nVolHeight * fluid->nVolWidth));
	if ( cudaMalloc ((void**)&dev_data, sizeof(unsigned char) * (fluid->nVolDepth * fluid->nVolHeight * fluid->nVolWidth)) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!" );
		return SG_RUNTIME_FALSE;
	}

	// Finally
	return SG_RUNTIME_OK;
}  

void FluidSimProc::FreeResourcePtrs ( void )
{
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( host_list [ i ] ) SAFE_FREE_PTR ( host_list [ i ] );
	}
	host_list.empty ( );

	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );

	SAFE_FREE_PTR (host_data);
	cudaFree (dev_data);
}

void FluidSimProc::ZeroData ( void )
{
	for ( int i = 0; i < Simul_Size; i++ )
	{
		host_u [ i ] = 0;
		host_v [ i ] = 0;
		host_w [ i ] = 0;
		host_den [ i ] = 0;
		host_den0 [ i ] = 0;
	}

	cudaDeviceDim3D();

	if ( cudaMemcpy (dev_u, host_u, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u0, host_u, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, host_v, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v0, host_v, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, host_w, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w0, host_w, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den, host_den, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den0, host_den, sizeof(int) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed" );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
}