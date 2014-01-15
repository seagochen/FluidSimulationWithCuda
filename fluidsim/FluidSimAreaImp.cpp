/**
* <Author>      Orlando Chen
* <First>       Nov 15, 2013
* <Last>		Jan 07, 2014
* <File>        FluidSimAreaImp.cpp
*/

#include <iostream>
#include <cuda_runtime.h>
#include "FluidSimArea.h"

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
	ZeroData ();
};

sge::SGRUNTIMEMSG sge::FluidSimProc::AllocateResourcePtrs ( fluidsim *fluid )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );

	// Allocate memory on host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static double *ptr;
		ptr = (double*) malloc ( Sim_Size * sizeof(double) );
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
		if ( cudaMalloc( (void **) &ptr, Sim_Size * sizeof(double) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	size_t size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;
	
	host_visual = (uchar*) malloc ( sizeof(uchar) * size );

	if ( cudaMalloc ((void**)&dev_visual, sizeof(uchar) * size ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}

	// Finally
	return SG_RUNTIME_OK;
}  

void sge::FluidSimProc::FreeResourcePtrs ( void )
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

	SAFE_FREE_PTR( host_visual );
	cudaFree ( dev_visual );
}

void sge::FluidSimProc::ZeroData ( void )
{
	for ( int i = 0; i < HostListNum; i++ )
	{
		for ( int j = 0; j < Sim_Size; j++ ) host_list[i][j] = 0.f;
		if ( cudaMemcpy (dev_list[i], host_list[i], 
			sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
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


void sge::FluidSimProc::CopyDataToDevice ( void )
{
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( cudaMemcpy (dev_list[i], host_list[i], 
			sizeof(double) * Sim_Size, cudaMemcpyHostToDevice) != cudaSuccess )
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


void sge::FluidSimProc::CopyDataToHost ( void )
{
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( cudaMemcpy (host_list[i], dev_list[i], 
			sizeof(double) * Sim_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
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