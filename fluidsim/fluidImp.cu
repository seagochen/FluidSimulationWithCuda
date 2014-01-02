#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "fluidsim.h"
#include "myMath.h"

using namespace sge;


__global__ void kernelPickData ( unsigned char *data, double const *grid )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] = (unsigned char) temp;
};

__global__ void kernelPickData ( unsigned char *data, double const *grid1, double const *grid2, double const *grid3 )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	// Add data from grid 1
	int temp = sground ( grid1[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	// Add data from grid 2
	temp += sground ( grid2[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;
	
	// Add data from grid 2
	temp += sground ( grid3[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] += (unsigned char) temp;
};


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
		host_u [ i ] = 0.f;
		host_v [ i ] = 0.f;
		host_w [ i ] = 0.f;
		host_den [ i ] = 0.f;
		host_div [ i ] = 0.f;
		host_p [ i ] = 0.f;
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
	if ( cudaMemcpy (dev_div, host_div, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_p, host_p, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
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
	if ( cudaMemcpy (dev_div, host_div, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_p, host_p, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
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
	if ( cudaMemcpy (host_div, dev_div, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_p, dev_p, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResourcePtrs ();
	exit(1);

Success:
	;	
};

void FluidSimProc::FluidSimSolver ( fluidsim *fluid )
{
	if ( !fluid->drawing.bContinue ) return ;

	// For fluid simulation, copy the data to device
	CopyDataToDevice();

	// Fluid process
	VelocitySolver ();
	DensitySolver ();
	PickData ( fluid );

	// Synchronize the device
	if ( cudaDeviceSynchronize() != cudaSuccess ) goto Error;

	// After simulation process, retrieve data back to host, in order to 
	// avoid data flipping
	CopyDataToHost();

	goto Success;

Error:
	cudaCheckErrors ("cudaDeviceSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	fluid->volume.ptrData = host_data;
};

void FluidSimProc::PickData ( fluidsim *fluid )
{
	cudaDeviceDim3D ();
	kernelPickData  <<<gridDim, blockDim>>> ( dev_data, dev_den );

	if ( cudaMemcpy (host_data, dev_data, 
		sizeof(unsigned char) * (fluid->volume.nVolDepth * fluid->volume.nVolHeight * fluid->volume.nVolWidth), 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed", __FILE__, __LINE__);
		FreeResourcePtrs ();
		exit (1);
	}
};