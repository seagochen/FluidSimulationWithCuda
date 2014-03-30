/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 30, 2014
* <File Name>     Kernels.h
*/

#ifndef __kernels_h__
#define __kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"


extern // updated: 2014/3/28
__global__ void kernelUpScalingInterpolation( double *dst, cdouble *src, 
						cint srcx, cint srcy, cint srcz,
						cint dstx, cint dsty, cint dstz,
						cint zoomx, cint zoomy, cint zoomz );


extern // updated: 2014/3/27
__global__ void kernelJacobi( double *out, cdouble *in, 
							cint tx, cint ty, cint tz,
							cdouble diffusion, cdouble divisor );


extern // updated: 2014/3/27
__global__ void kernelAdvection( double *out, cdouble *in, 
							cint tx, cint ty, cint tz,
							cdouble delta, cdouble *u, cdouble *v, cdouble *w );


extern  // updated: 2014/3/27
__global__ void kernelGradient( double *div, double *prs,
							cint tx, cint ty, cint tz,
							cdouble *u, cdouble *v, cdouble *w );


extern // updated: 2014/3/27
__global__ void kernelSubtract( double *u, double *v, double *w, double *prs,
							cint tx, cint ty, cint tz );


extern // updated: 2014/3/27
__global__ void kernelAddSource( double *dens, double *v,
							cint tx, cint ty, cint tz,
							cdouble *obst, cdouble dtime, cdouble rate );


extern // updated: 2014/3/30
__global__ void kernelSetBound( double *dst, cint dstx, cint dsty, cint dstz );


extern // updated" 2014/3/30
__global__ void kernelPickData( uchar *volume, cdouble *src, 
		cint dstx, cint dsty, cint dstz,
		cint srcx, cint srcy, cint srcz,
		cint offi, cint offj, cint offk );


extern // updated: 2014/3/28
__global__ void kernelPickData( uchar *volume, cdouble *src, cint tx, cint ty, cint tz );


extern // updated: 2014/3/30
__global__ void kernelAssembleCompBufs( double *dst, cdouble *src,
		cint srcx, cint srcy, cint srcz, 
		cint dstx, cint dsty, cint dstz,
		cint offi, cint offj, cint offk );


extern // updated: 2014/3/30
__global__ void kernelDeassembleCompBufs( double *dst, cdouble *src,
		cint srcx, cint srcy, cint srcz,
		cint dstx, cint dsty, cint dstz,
		cint offi, cint offj, cint offk	);


extern // updated: 2014/3/30
__global__ void kernelFillBullet( double *dst, cdouble *src,
		cint srcx, cint srcy, cint srcz, 
		cint dstx, cint dsty, cint dstz, 
		cint grdx, cint grdy, cint grdz,
		cint offi, cint offj, cint offk );


extern __global__ void kernelLoadBullet
	( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz );


extern __global__ void kernelLoadBullet
	( double *dst, cdouble *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz );


extern __global__ void kernelExitBullet
	( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz );


extern __global__ void kernelExitBullet
	( double *dst, cdouble *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz );



extern __global__ void kernelZeroBuffers( int *bullet, cint tilex, cint tiley, cint tilez );


extern __global__ void kernelZeroBuffers( double *bullet, cint tilex, cint tiley, cint tilez );


extern __global__ void kernelZeroBuffers( uchar *bullet, cint tilex, cint tiley, cint tilez );


extern __global__ void kernelZeroBuffers( int *buf, cint tiles );


extern __global__ void kernelZeroBuffers( double *buf, cint tiles );


extern __global__ void kernelZeroBuffers( uchar *buf, cint tiles );


extern __global__ void kernelCopyBuffers( int *dst, cint *src, cint tiles );


extern __global__ void kernelCopyBuffers( double *dst, cint *src, cint tiles );


extern __global__ void kernelCopyBuffers( uchar *dst, cint *src, cint tiles );


extern __global__ void kernelCopyBuffers( int *dst, cint *src, cint tilex, cint tiley, cint tilez );


extern __global__ void kernelCopyBuffers( double *dst, cdouble *src, cint tilex, cint tiley, cint tilez );


extern __global__ void kernelCopyBuffers( uchar *dst, uchar *src, cint tilex, cint tiley, cint tilez );

#endif