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

__device__ void atomicTracingUp( double *up, double const *center )
{
	GetIndex();

	up[Index(i,sim_header,k)] = center[Index(i,sim_tailer,k)];

	up[Index(sim_header,sim_header,k)] = center[Index(sim_header,sim_tailer,k)] / 2.f;
	up[Index(sim_tailer,sim_header,k)] = center[Index(sim_tailer,sim_tailer,k)] / 2.f;
	up[Index(i,sim_header,sim_header)] = center[Index(i,sim_tailer,sim_header)] / 2.f;
	up[Index(i,sim_header,sim_tailer)] = center[Index(i,sim_tailer,sim_tailer)] / 2.f;

	up[Index(sim_header,sim_header,sim_header)] = center[Index(sim_header,sim_tailer,sim_header)] / 3.f;
	up[Index(sim_tailer,sim_header,sim_header)] = center[Index(sim_tailer,sim_tailer,sim_header)] / 3.f;
	up[Index(sim_header,sim_header,sim_tailer)] = center[Index(sim_header,sim_tailer,sim_tailer)] / 3.f;
	up[Index(sim_tailer,sim_header,sim_tailer)] = center[Index(sim_tailer,sim_tailer,sim_tailer)] / 3.f;
};

__device__ void atomicTracingDown( double *down, double const *center )
{
	GetIndex();

	down[Index(i,sim_tailer,k)] = center[Index(i,sim_header,k)];

	down[Index(sim_header,sim_tailer,k)] = center[Index(sim_header,sim_header,k)] / 2.f;
	down[Index(sim_tailer,sim_tailer,k)] = center[Index(sim_tailer,sim_header,k)] / 2.f;
	down[Index(i,sim_tailer,sim_header)] = center[Index(i,sim_header,sim_header)] / 2.f;
	down[Index(i,sim_tailer,sim_tailer)] = center[Index(i,sim_header,sim_tailer)] / 2.f;

	down[Index(sim_header,sim_tailer,sim_header)] = center[Index(sim_header,sim_header,sim_header)] / 3.f;
	down[Index(sim_tailer,sim_tailer,sim_header)] = center[Index(sim_tailer,sim_header,sim_header)] / 3.f;
	down[Index(sim_header,sim_tailer,sim_tailer)] = center[Index(sim_header,sim_header,sim_tailer)] / 3.f;
	down[Index(sim_tailer,sim_tailer,sim_tailer)] = center[Index(sim_tailer,sim_header,sim_tailer)] / 3.f;
};

__device__ void atomicTracingLeft( double *left, double const *center )
{
	GetIndex();

	left[Index(sim_tailer,j,k)] = center[Index(sim_header,j,k)];

	left[Index(sim_tailer,sim_header,k)] = center[Index(sim_header,sim_header,k)] / 2.f;
	left[Index(sim_tailer,sim_tailer,k)] = center[Index(sim_header,sim_tailer,k)] / 2.f;
	left[Index(sim_tailer,j,sim_header)] = center[Index(sim_header,j,sim_header)] / 2.f;
	left[Index(sim_tailer,j,sim_tailer)] = center[Index(sim_header,j,sim_tailer)] / 2.f;

	left[Index(sim_tailer,sim_header,sim_header)] = center[Index(sim_header,sim_header,sim_header)] / 3.f; 
	left[Index(sim_tailer,sim_header,sim_tailer)] = center[Index(sim_header,sim_header,sim_tailer)] / 3.f;
	left[Index(sim_tailer,sim_tailer,sim_header)] = center[Index(sim_header,sim_tailer,sim_header)] / 3.f;
	left[Index(sim_tailer,sim_tailer,sim_tailer)] = center[Index(sim_header,sim_tailer,sim_tailer)] / 3.f;
};

__device__ void atomicTracingRight( double *right, double const *center )
{
	GetIndex();

	right[Index(sim_header,j,k)] = center[Index(sim_tailer,j,k)];

	right[Index(sim_header,sim_header,k)] = center[Index(sim_tailer,sim_header,k)] / 2.f;
	right[Index(sim_header,sim_tailer,k)] = center[Index(sim_tailer,sim_tailer,k)] / 2.f;
	right[Index(sim_header,j,sim_header)] = center[Index(sim_tailer,j,sim_header)] / 2.f;
	right[Index(sim_header,j,sim_tailer)] = center[Index(sim_tailer,j,sim_tailer)] / 2.f;

	right[Index(sim_header,sim_header,sim_header)] = center[Index(sim_tailer,sim_header,sim_header)] / 3.f;
	right[Index(sim_header,sim_header,sim_tailer)] = center[Index(sim_tailer,sim_header,sim_tailer)] / 3.f;
	right[Index(sim_header,sim_tailer,sim_header)] = center[Index(sim_tailer,sim_tailer,sim_header)] / 3.f;
	right[Index(sim_header,sim_tailer,sim_tailer)] = center[Index(sim_tailer,sim_tailer,sim_tailer)] / 3.f;
};

__device__ void atomicTracingFront( double *front, double const *center )
{
	GetIndex();

	front[Index(i,j,sim_header)] = center[Index(i,j,sim_tailer)];
	
	front[Index(sim_header,j,sim_header)] = center[Index(sim_header,j,sim_tailer)] / 2.f;
	front[Index(sim_tailer,j,sim_header)] = center[Index(sim_tailer,j,sim_tailer)] / 2.f;
	front[Index(i,sim_header,sim_header)] = center[Index(i,sim_header,sim_tailer)] / 2.f;
	front[Index(i,sim_tailer,sim_header)] = center[Index(i,sim_tailer,sim_tailer)] / 2.f;

	front[Index(sim_header,sim_header,sim_header)] = center[Index(sim_header,sim_header,sim_tailer)] / 3.f;
	front[Index(sim_header,sim_tailer,sim_header)] = center[Index(sim_header,sim_tailer,sim_tailer)] / 3.f;
	front[Index(sim_tailer,sim_header,sim_header)] = center[Index(sim_tailer,sim_header,sim_tailer)] / 3.f;
	front[Index(sim_tailer,sim_tailer,sim_header)] = center[Index(sim_tailer,sim_tailer,sim_tailer)] / 3.f;
};

__device__ void atomicTracingBack( double *back, double const *center )
{
	GetIndex();

	back[Index(i,j,sim_tailer)] = center[Index(i,j,sim_header)];
	
	back[Index(sim_header,j,sim_tailer)] = center[Index(sim_header,j,sim_header)] / 2.f;
	back[Index(sim_tailer,j,sim_tailer)] = center[Index(sim_tailer,j,sim_header)] / 2.f;
	back[Index(i,sim_header,sim_tailer)] = center[Index(i,sim_header,sim_header)] / 2.f;
	back[Index(i,sim_tailer,sim_tailer)] = center[Index(i,sim_tailer,sim_header)] / 2.f;

	back[Index(sim_header,sim_header,sim_tailer)] = center[Index(sim_header,sim_header,sim_header)] / 3.f;
	back[Index(sim_header,sim_tailer,sim_tailer)] = center[Index(sim_header,sim_tailer,sim_header)] / 3.f;
	back[Index(sim_tailer,sim_header,sim_tailer)] = center[Index(sim_tailer,sim_header,sim_header)] / 3.f;
	back[Index(sim_tailer,sim_tailer,sim_tailer)] = center[Index(sim_tailer,sim_tailer,sim_header)] / 3.f;
};

__global__ void kernelFloodingBuffers
	( double *left, double *right, double *up, double *down, double *front, double *back, double *center )
{
	atomicTracingUp(       up, center );
	atomicTracingDown(   down, center );
	atomicTracingLeft(   left, center );
	atomicTracingRight( right, center );
	atomicTracingFront( front, center );
	atomicTracingBack (  back, center );
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
	( double *bufs, double *left, double *right, double *up, double *down, double *front, double *back, double *center )
{
	GetIndex();

	bufs[TEMP_BUF_CENTER] += center[Index(i,j,k)];
	bufs[TEMP_BUF_LEFT]   +=   left[Index(i,j,k)];
	bufs[TEMP_BUF_RIGHT]  +=  right[Index(i,j,k)];
	bufs[TEMP_BUF_UP]     +=     up[Index(i,j,k)];
	bufs[TEMP_BUF_DOWN]   +=   down[Index(i,j,k)];
	bufs[TEMP_BUF_FRONT]  +=  front[Index(i,j,k)];
	bufs[TEMP_BUF_BACK]   +=   back[Index(i,j,k)];
};

#endif