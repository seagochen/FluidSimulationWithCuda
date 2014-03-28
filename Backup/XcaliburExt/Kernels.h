/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 26, 2014
* <File Name>     Kernels.h
*/

#ifndef __kernels_h__
#define __kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"

// updated: 2014/3/27
extern __global__ void kernelJacobi( double *out, cdouble *in, 
							 cint tx, cint ty, cint tz,
							 cdouble diffusion, cdouble divisor );

// updated: 2014/3/27
extern __global__ void kernelAdvection( double *out, cdouble *in, 
								cint tx, cint ty, cint tz,
								cdouble delta, cdouble *u, cdouble *v, cdouble *w );

// updated: 2014/3/27
extern __global__ void kernelGradient( double *div, double *prs,
							   cint tx, cint ty, cint tz,
							   cdouble *u, cdouble *v, cdouble *w );

// updated: 2014/3/27
extern __global__ void kernelSubtract( double *u, double *v, double *w, double *prs,
							   cint tx, cint ty, cint tz );


// updated: 2014/3/27
extern __global__ void kernelAddSource( double *dens, double *v,
								cint tx, cint ty, cint tz,
								cdouble *obst, cdouble dtime, cdouble rate );

// updated: 2014/3/27
extern __global__ void kernelPickData( uchar *volume, cint dstx, cint dsty, cint dstz,
							   cdouble *src, cint srcx, cint srcy, cint srcz,
							   cint offi, cint offj, cint offk, 
							   cdouble zoomx, cdouble zoomy, cdouble zoomz );

// updated: 2014/3/27
extern __global__ void kernelPickData( uchar *volume, cdouble *src,
									  cint tx, cint ty, cint tz );


// updated: 2014/3/27
extern __global__ void kernelAssembleCompBufs( double *dst,  cint dstx, cint dsty, cint dstz, 
									  cdouble *src,  cint srcx, cint srcy, cint srcz,
									  cint offi, cint offj, cint offk, 
									  cdouble zoomx, cdouble zoomy, cdouble zoomz );

// updated: 2014/3/27
extern __global__ void kernelDeassembleCompBufs( double *dst,  cint dstx, cint dsty, cint dstz, 
										 cdouble *src,  cint srcx, cint srcy, cint srcz,
										 cint offi, cint offj, cint offk, 
										 cdouble zoomx, cdouble zoomy, cdouble zoomz );

#if 0
extern __global__ void kernelJacobi( double *out, cdouble *in, cdouble diffusion, cdouble divisor );

extern __global__ void kernelAdvection( double *out, cdouble *in, cdouble delta, cdouble *u, cdouble *v, cdouble *w );

extern __global__ void kernelGradient( double *div, double *prs, cdouble *u, cdouble *v, cdouble *w );

extern __global__ void kernelSubtract( double *u, double *v, double *w, double *prs );

extern __global__ void kernelAddSource( double *dens, double *v, cdouble *obst, cdouble dtime, cdouble rate );

extern __global__ void kernelPickData
	( uchar *volume, cdouble *rho, int offi, int offj, int offk, cint gridx, cint gridy, cint gridz );

extern __global__ void kernelInterRootGrids( double *dst, cdouble *src, cint i, cint j, cint k, cdouble rate );

extern __global__ void kernelInterLeafGrids( double *dst, cdouble *src, cint i, cint j, cint k, cdouble rate );
#endif



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



extern __global__ void kernelLoadLeftFace( double *bullet, cdouble *face,
										  cint dstx, cint dsty, cint dstz, 
										  cint srcx, cint srcy, cint srcz );

extern __global__ void kernelLoadRightFace( double *bullet, cdouble *face,
										  cint dstx, cint dsty, cint dstz, 
										  cint srcx, cint srcy, cint srcz );

extern __global__ void kernelLoadUpFace( double *bullet, cdouble *face,
										  cint dstx, cint dsty, cint dstz, 
										  cint srcx, cint srcy, cint srcz );

extern __global__ void kernelLoadDownFace( double *bullet, cdouble *face,
										  cint dstx, cint dsty, cint dstz, 
										  cint srcx, cint srcy, cint srcz );

extern __global__ void kernelLoadFrontFace( double *bullet, cdouble *face,
										  cint dstx, cint dsty, cint dstz, 
										  cint srcx, cint srcy, cint srcz );

extern __global__ void kernelLoadBackFace( double *bullet, cdouble *face,
										  cint dstx, cint dsty, cint dstz, 
										  cint srcx, cint srcy, cint srcz );

extern __global__ void kernelSmoothBullet
	( double *bullet, cint bx, cint by, cint bz, cint gx, cint gy, cint gz );

#endif