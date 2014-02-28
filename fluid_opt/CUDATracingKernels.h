/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Feb 23, 2014
* <File Name>     CUDATracingKernels.h
*/

#ifndef __cuda_tracing_kernerls_h__
#define __cuda_tracing_kernerls_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "CUDAMacroDef.h"

#define TEMP_BUF_CENTER   0
#define TEMP_BUF_LEFT     1
#define TEMP_BUF_RIGHT    2
#define TEMP_BUF_UP       3
#define TEMP_BUF_DOWN     4
#define TEMP_BUF_FRONT    5
#define TEMP_BUF_BACK     6

__device__ void atomicTracingUp( double *center, const double *up )
{
	GetIndex();
	BeginSimArea();


	EndSimArea();
};

__device__ void atomicTracingDown( double *center, const double *down )
{
	GetIndex();
	BeginSimArea();


	EndSimArea();
};

__device__ void atomicTracingLeft( double *center, const double *left )
{
	GetIndex();
	BeginSimArea();
	

	EndSimArea();
};

__device__ void atomicTracingRight( double *center, const double *right )
{
	GetIndex();
	BeginSimArea();

	EndSimArea();
};

__device__ void atomicTracingFront( double *center, const double *front )
{
	GetIndex();
	BeginSimArea();
	

	EndSimArea();	
};

__device__ void atomicTracingBack( double *center, const double *back )
{
	GetIndex();
	BeginSimArea();


	EndSimArea();
};

__global__ void kernelFloodingBuffers
	( double *left, double *right, double *up, double *down, double *front, double *back, double *center )
{

};

__device__ void atomicClearHalo( double *grids )
{
	GetIndex();

	grids[Index(gst_header,j,k)] = 0.f;
	grids[Index(gst_tailer,j,k)] = 0.f;
	grids[Index(i,gst_header,k)] = 0.f;
	grids[Index(i,gst_tailer,k)] = 0.f;
	grids[Index(i,j,gst_header)] = 0.f;
	grids[Index(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelClearHalo
	( double *left, double *right, double *up, double *down, double *front, double *back, double *center )
{
	GetIndex();

	atomicClearHalo( left );
	atomicClearHalo( right );
	atomicClearHalo( up );
	atomicClearHalo( down );
	atomicClearHalo( front );
	atomicClearHalo( back );
};

__global__ void kernelSumBufsDens
	( double *bufs, 
	const double *left,  const double *right,
	const double *up,    const double *down,
	const double *front, const double *back,
	const double *center )
{
	GetIndex();

	bufs[TEMP_BUF_LEFT]   += (  left[Index(i,j,k)] > 0.f ) ?  left[Index(i,j,k)] : 0.f;
	bufs[TEMP_BUF_RIGHT]  += ( right[Index(i,j,k)] > 0.f ) ? right[Index(i,j,k)] : 0.f;
	bufs[TEMP_BUF_UP]     += (    up[Index(i,j,k)] > 0.f ) ?    up[Index(i,j,k)] : 0.f;
	bufs[TEMP_BUF_DOWN]   += (  down[Index(i,j,k)] > 0.f ) ?  down[Index(i,j,k)] : 0.f;
	bufs[TEMP_BUF_FRONT]  += ( front[Index(i,j,k)] > 0.f ) ? front[Index(i,j,k)] : 0.f;
	bufs[TEMP_BUF_BACK]   += (  back[Index(i,j,k)] > 0.f ) ?  back[Index(i,j,k)] : 0.f;
	bufs[TEMP_BUF_CENTER] += ( center[Index(i,j,k)] > 0.f ) ? center[Index(i,j,k)] : 0.f;


	bufs[TEMP_BUF_LEFT]   += ( center[Index(sim_header,j,k)] > 0.f ) ? center[Index(sim_header,j,k)] : 0.f;
	bufs[TEMP_BUF_RIGHT]  += ( center[Index(sim_tailer,j,k)] > 0.f ) ? center[Index(sim_tailer,j,k)] : 0.f;
	bufs[TEMP_BUF_UP]     += ( center[Index(i,sim_header,k)] > 0.f ) ? center[Index(i,sim_header,k)] : 0.f;
	bufs[TEMP_BUF_DOWN]   += ( center[Index(i,sim_tailer,k)] > 0.f ) ? center[Index(i,sim_tailer,k)] : 0.f;
	bufs[TEMP_BUF_FRONT]  += ( center[Index(i,j,sim_header)] > 0.f ) ? center[Index(i,j,sim_header)] : 0.f;
	bufs[TEMP_BUF_BACK]   += ( center[Index(i,j,sim_tailer)] > 0.f ) ? center[Index(i,j,sim_tailer)] : 0.f;
};

#endif