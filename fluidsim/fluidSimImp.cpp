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

	fluid->fps.dwCurrentTime = 0;
	fluid->fps.dwElapsedTime = 0;
	fluid->fps.dwFrames = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.FPS = 0;

	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	ZeroData ();
};

SGRUNTIMEMSG FluidSimProc::AllocateResourcePtrs ( fluidsim *fluid )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );

	// Allocate memory on host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static double *ptr;
		ptr = (double*) malloc ( Simul_Size * sizeof(double) );
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
		static double *ptr;
		if ( cudaMalloc( (void **) &ptr, Simul_Size * sizeof(double) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}
	
	host_data = (GLubyte*) malloc (sizeof(GLubyte) * 
		(fluid->volume.nVolDepth * fluid->volume.nVolHeight * fluid->volume.nVolWidth));
	if ( cudaMalloc ((void**)&dev_data, sizeof(unsigned char) * 
		(fluid->volume.nVolDepth * fluid->volume.nVolHeight * fluid->volume.nVolWidth)) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
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

	if ( cudaMemcpy (dev_u, host_u, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u0, host_u, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, host_v, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v0, host_v, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, host_w, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w0, host_w, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den, host_den, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den0, host_den, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;
}


void FluidSimProc::CopyDataToDevice ( void )
{
	if ( cudaMemcpy (dev_u, host_u, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, host_v, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, host_w, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den, host_den, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};


void FluidSimProc::CopyDataToHost ( void )
{
	if ( cudaMemcpy (host_u, dev_u, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_v, dev_v, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_w, dev_w, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_den, dev_den, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};