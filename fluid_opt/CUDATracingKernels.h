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

__global__ void kernelFloodingBuffers
	( double *left, double *right, double *up, double *down, double *front, double *back, double *center )
{
	GetIndex();
	center[Index(sim_header,j,k)] = ( left[Index(sim_tailer,j,k)] + center[Index(sim_header,j,k)] ) / 2.f;
	center[Index(sim_tailer,j,k)] = ( right[Index(sim_header,j,k)] + center[Index(sim_tailer,j,k)] ) / 2.f;
	center[Index(i,sim_tailer,k)] = ( up[Index(i,sim_header,k)] + center[Index(i,sim_tailer,k)] ) / 2.f;
	center[Index(i,sim_header,k)] = ( down[Index(i,sim_tailer,k)] + center[Index(i,sim_header,k)] ) / 2.f;
	center[Index(i,j,sim_tailer)] = ( front[Index(i,j,sim_header)] + center[Index(i,j,sim_tailer)] ) / 2.f;
	center[Index(i,j,sim_header)] = ( back[Index(i,j,sim_tailer)] + center[Index(i,j,sim_header)] ) / 2.f;
};

__global__ void kernelFloodBound( unsigned char *grids, int offi, int offj, int offk )
{
	GetIndex();

	i = offi * GRIDS_X + i;
	j = offj * GRIDS_X + j;
	k = offk * GRIDS_X + k;

	/* faces */
	grids[Index(gst_header,j,k)] = grids[Index(sim_header,j,k)];
	grids[Index(gst_tailer,j,k)] = grids[Index(sim_tailer,j,k)];
	grids[Index(i,gst_header,k)] = grids[Index(i,sim_header,k)];
	grids[Index(i,gst_tailer,k)] = grids[Index(i,sim_tailer,k)];
	grids[Index(i,j,gst_header)] = grids[Index(i,j,sim_header)];
	grids[Index(i,j,gst_tailer)] = grids[Index(i,j,sim_tailer)];

	/* edges */
	grids[Index(i,gst_header,gst_header)] = ( grids[Index(i,sim_header,gst_header)] + grids[Index(i,gst_header,sim_header)] ) / 2.f;
	grids[Index(i,gst_tailer,gst_header)] = ( grids[Index(i,sim_tailer,gst_header)] + grids[Index(i,gst_tailer,sim_header)] ) / 2.f;
	grids[Index(i,gst_header,gst_tailer)] = ( grids[Index(i,sim_header,gst_tailer)] + grids[Index(i,gst_header,sim_tailer)] ) / 2.f;
	grids[Index(i,gst_tailer,gst_tailer)] = ( grids[Index(i,sim_tailer,gst_tailer)] + grids[Index(i,gst_tailer,sim_tailer)] ) / 2.f;

	grids[Index(gst_header,j,gst_header)] = ( grids[Index(sim_header,j,gst_header)] + grids[Index(gst_header,j,sim_header)] ) / 2.f;
	grids[Index(gst_tailer,j,gst_header)] = ( grids[Index(sim_tailer,j,gst_header)] + grids[Index(gst_tailer,j,sim_header)] ) / 2.f;
	grids[Index(gst_header,j,gst_tailer)] = ( grids[Index(sim_header,j,gst_tailer)] + grids[Index(gst_header,j,sim_tailer)] ) / 2.f;
	grids[Index(gst_tailer,j,gst_tailer)] = ( grids[Index(sim_tailer,j,gst_tailer)] + grids[Index(gst_tailer,j,sim_tailer)] ) / 2.f;

	grids[Index(gst_header,gst_header,k)] = ( grids[Index(sim_header,gst_header,k)] + grids[Index(gst_header,sim_header,k)] ) / 2.f;
	grids[Index(gst_tailer,gst_header,k)] = ( grids[Index(sim_tailer,gst_header,k)] + grids[Index(gst_tailer,sim_header,k)] ) / 2.f;
	grids[Index(gst_header,gst_tailer,k)] = ( grids[Index(sim_header,gst_tailer,k)] + grids[Index(gst_header,sim_tailer,k)] ) / 2.f;
	grids[Index(gst_tailer,gst_tailer,k)] = ( grids[Index(sim_tailer,gst_tailer,k)] + grids[Index(gst_tailer,sim_tailer,k)] ) / 2.f;

	/* vetices */
	grids[Index(gst_header,gst_header,gst_header)] = ( grids[Index(sim_header,gst_header,gst_header)] + grids[Index(gst_header,sim_header,gst_header)] + grids[Index(gst_header,gst_header,sim_header)] ) / 3.f;
	grids[Index(gst_header,gst_header,gst_tailer)] = ( grids[Index(sim_header,gst_header,gst_tailer)] + grids[Index(gst_header,sim_header,gst_tailer)] + grids[Index(gst_header,gst_header,sim_tailer)] ) / 3.f;
	grids[Index(gst_header,gst_tailer,gst_header)] = ( grids[Index(sim_header,gst_tailer,gst_header)] + grids[Index(gst_header,sim_tailer,gst_header)] + grids[Index(gst_header,gst_tailer,sim_header)] ) / 3.f;
	grids[Index(gst_header,gst_tailer,gst_tailer)] = ( grids[Index(sim_header,gst_tailer,gst_tailer)] + grids[Index(gst_header,sim_tailer,gst_tailer)] + grids[Index(gst_header,gst_tailer,sim_tailer)] ) / 3.f;
	grids[Index(gst_tailer,gst_header,gst_header)] = ( grids[Index(sim_tailer,gst_header,gst_header)] + grids[Index(gst_tailer,sim_header,gst_header)] + grids[Index(gst_tailer,gst_header,sim_header)] ) / 3.f;
	grids[Index(gst_tailer,gst_header,gst_tailer)] = ( grids[Index(sim_tailer,gst_header,gst_tailer)] + grids[Index(gst_tailer,sim_header,gst_tailer)] + grids[Index(gst_tailer,gst_header,sim_tailer)] ) / 3.f;
	grids[Index(gst_tailer,gst_tailer,gst_header)] = ( grids[Index(sim_tailer,gst_tailer,gst_header)] + grids[Index(gst_tailer,sim_tailer,gst_header)] + grids[Index(gst_tailer,gst_tailer,sim_header)] ) / 3.f;
	grids[Index(gst_tailer,gst_tailer,gst_tailer)] = ( grids[Index(sim_tailer,gst_tailer,gst_tailer)] + grids[Index(gst_tailer,sim_tailer,gst_tailer)] + grids[Index(gst_tailer,gst_tailer,sim_tailer)] ) / 3.f;
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