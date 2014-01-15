/**
* <Author>      Orlando Chen
* <First>       Oct 10, 2013
* <Last>		Jan 15, 2014
* <File>        BufferOperationDynamic.h
*/

#ifndef __buffer_operation_dynamic_h_
#define __buffer_operation_dynamic_h_

#include "FluidSimAreaDynamic.h"
#include "FluidMathLibDynamic.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ 
void kernelPickData ( unsigned char *data, double const *grid, 
	int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};

__global__ 
void kernelCopyBuffer ( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ 
void kernelSwapBuffer ( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__host__ 
void hostSwapBuffer ( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
};

__global__
void kernelZeroBuffer ( double *grid )
{
	GetIndex();
	grid[ Index(i,j,k) ] = 0.f;
};

__global__
void kernelZeroBuffer ( unsigned char *grid, int const offi, int const offj, int const offk )
{
	GetIndex();
	int di = offi + i;
	int dj = offj + j;
	int dk = offk + k;
	grid [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;	
};

__host__
void hostZeroBuffer ( double *grid )
{
	cudaDeviceDim3D();
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( grid );
};

__host__
void hostZeroBuffer ( unsigned char *grid, int const offi, int const offj, int const offk )
{
	cudaDeviceDim3D();
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( grid, offi, offj, offk );
};

__device__ 
double i0j0k0 ( double *grid )
{
	double temp =
		grid [ Index(gst_header, gst_header, gst_header) ] + 
		grid [ Index(sim_header, gst_header, gst_header) ] + 
		grid [ Index(gst_header, sim_header, gst_header) ] +
		grid [ Index(gst_header, gst_header, sim_header) ];
	return temp / 4.f;
};

__device__ 
double i1j0k0 ( double *grid )
{
	double temp =
		grid [ Index(gst_tailer, gst_header, gst_header) ] + 
		grid [ Index(sim_tailer, gst_header, gst_header) ] + 
		grid [ Index(gst_tailer, sim_header, gst_header) ] +
		grid [ Index(gst_tailer, gst_header, sim_header) ];
	return temp / 4.f;
};

__device__ 
double i0j1k0 ( double *grid )
{
	double temp = 0.f;
	temp = grid [ Index(gst_header, gst_tailer, gst_header) ] + 
		grid [ Index(gst_header, sim_tailer, gst_header) ] + 
		grid [ Index(sim_header, gst_tailer, gst_header) ] +
		grid [ Index(gst_header, gst_tailer, sim_header) ];
	return temp / 4.f;
};

__device__ 
double i1j1k0 ( double *grid )
{
	double temp = 0.f;
	temp = grid [ Index(gst_tailer, gst_tailer, gst_header) ] + 
		grid [ Index(sim_tailer, gst_tailer, gst_header) ] + 
		grid [ Index(gst_tailer, sim_tailer, gst_header) ] +
		grid [ Index(gst_tailer, gst_tailer, sim_header) ];
	return temp / 4.f;
};

__device__ 
double i0j0k1 ( double *grid )
{
	double temp = 0.f;
	temp = grid [ Index(gst_header, gst_header, gst_tailer) ] +
		grid [ Index(gst_header, gst_header, sim_tailer) ] +
		grid [ Index(gst_header, sim_header, gst_tailer) ] + 
		grid [ Index(sim_header, gst_header, gst_tailer) ];
	return temp / 4.f;
};

__device__ 
double i1j0k1 ( double *grid )
{
	double temp = 0.f;
	temp = grid [ Index(gst_tailer, gst_header, gst_tailer) ] +
		grid [ Index(sim_tailer, gst_header, gst_tailer) ] + 
		grid [ Index(gst_tailer, sim_header, gst_tailer) ] + 
		grid [ Index(gst_tailer, gst_header, sim_tailer) ];
	return temp / 4.f;
};

__device__ 
double i0j1k1 ( double *grid )
{
	double temp = 0.f;
	temp = grid [ Index(gst_header, gst_tailer, gst_tailer) ] +
		grid [ Index(gst_header, gst_tailer, sim_tailer) ] + 
		grid [ Index(gst_header, sim_tailer, gst_tailer) ] +
		grid [ Index(sim_header, gst_tailer, gst_tailer) ];
	return temp / 4.f;
};

__device__ 
double i1j1k1 ( double *grid )
{
	double temp = 0.f;
	temp = grid [ Index(gst_tailer, gst_tailer, gst_tailer) ] +
		grid [ Index(sim_tailer, gst_tailer, gst_tailer) ] +
		grid [ Index(gst_tailer, sim_tailer, gst_tailer) ] +
		grid [ Index(gst_tailer, gst_tailer, sim_tailer) ];
	return temp / 4.f;
};

#endif