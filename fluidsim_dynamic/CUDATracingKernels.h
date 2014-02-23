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

__device__ void atomicHandleFaceLR( double *grids, double const *center, int header, int tailer )
{
	GetIndex();
	
	/* copy entire face from center node */
	grids[Index(header,j,k)] = center[Index(tailer,j,k)];

	/* wait, we have left something which may cause un-conservation of mass */
	grids[Index(header,sim_header,k)] = center[Index(tailer,sim_header,k)] / 2.f;
	grids[Index(header,sim_tailer,k)] = center[Index(tailer,sim_tailer,k)] / 2.f;
	grids[Index(header,j,sim_header)] = center[Index(tailer,j,sim_header)] / 2.f;
	grids[Index(header,j,sim_tailer)] = center[Index(tailer,j,sim_tailer)] / 2.f;

	grids[Index(header,sim_header,sim_header)] = center[Index(tailer,sim_header,sim_header)] / 3.f;
	grids[Index(header,sim_header,sim_tailer)] = center[Index(tailer,sim_header,sim_tailer)] / 3.f;
	grids[Index(header,sim_tailer,sim_header)] = center[Index(tailer,sim_tailer,sim_header)] / 3.f;
	grids[Index(header,sim_tailer,sim_tailer)] = center[Index(tailer,sim_tailer,sim_tailer)] / 3.f;
};
 
__device__ void atomicHandleFaceUD( double *grids, double const *center, int header, int tailer )
{
	GetIndex();

	grids[Index(i,header,k)] = center[Index(i,tailer,k)];

	grids[Index(sim_header,header,k)] = center[Index(sim_header,tailer,k)] / 2.f;
	grids[Index(sim_tailer,header,k)] = center[Index(sim_tailer,tailer,k)] / 2.f;
	grids[Index(i,header,sim_header)] = center[Index(i,tailer,sim_header)] / 2.f;
	grids[Index(i,header,sim_tailer)] = center[Index(i,tailer,sim_tailer)] / 2.f;

	grids[Index(sim_header,header,sim_header)] = center[Index(sim_header,tailer,sim_header)] / 2.f;
	grids[Index(sim_tailer,header,sim_header)] = center[Index(sim_tailer,tailer,sim_header)] / 2.f;
	grids[Index(sim_header,header,sim_tailer)] = center[Index(sim_header,tailer,sim_tailer)] / 2.f;
	grids[Index(sim_tailer,header,sim_tailer)] = center[Index(sim_tailer,tailer,sim_tailer)] / 2.f;
};

__device__ void atomicHandleFaceFB( double *grids, double const *center, int header, int tailer )
{
	GetIndex();
 
	grids[Index(i,j,header)] = center[Index(i,j,tailer)];
 
	grids[Index(sim_header,j,header)] = center[Index(sim_header,j,tailer)] / 2.f;
	grids[Index(i,sim_header,header)] = center[Index(i,sim_header,tailer)] / 2.f;
	grids[Index(sim_tailer,j,header)] = center[Index(sim_tailer,j,tailer)] / 2.f;
	grids[Index(i,sim_tailer,header)] = center[Index(i,sim_tailer,tailer)] / 2.f;

	grids[Index(sim_header,sim_header,header)] = center[Index(sim_header,sim_header,tailer)] / 3.f;
	grids[Index(sim_header,sim_tailer,header)] = center[Index(sim_header,sim_tailer,tailer)] / 3.f;
	grids[Index(sim_tailer,sim_header,header)] = center[Index(sim_tailer,sim_header,tailer)] / 3.f;
	grids[Index(sim_tailer,sim_tailer,header)] = center[Index(sim_tailer,sim_tailer,tailer)] / 3.f;
};


__global__ void kernelFloodingBuffers
	( double *left, double *right, double *up, double *down, double *front, double *back, double *center )
{
	atomicHandleFaceLR( left,  center, sim_tailer, sim_header );
	atomicHandleFaceLR( right, center, sim_header, sim_tailer );
	atomicHandleFaceUD( up,    center, sim_header, sim_tailer );
	atomicHandleFaceUD( down,  center, sim_tailer, sim_header );
	atomicHandleFaceFB( front, center, sim_header, sim_tailer );
	atomicHandleFaceFB( back,  center, sim_tailer, sim_header );
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