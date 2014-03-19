/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 19, 2014
* <File Name>     Kernels.h
*/

#ifndef __kernels_h__
#define __kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

typedef int const     cint;
typedef unsigned char uchar;
typedef double const  cdouble;

extern __global__ void kernelExitBullet( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz );

extern __global__ void kernelExitBullet( double *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz );

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




extern __global__ void kernelObstacle( double *grids, cdouble *obstacle, cint field, cint tx, cint ty, cint tz );

extern __global__ void kernelJacobi
	( double *grid_out, cdouble *grid_in, cdouble diffusion, cdouble divisor, cint tx, cint ty, cint tz );

extern __global__ void kernelGridAdvection
	( double *grid_out, cdouble *grid_in, cdouble deltatime, cdouble *u_in, cdouble *v_in, cdouble *w_in,
	cint tx, cint ty, cint tz );

extern __global__ void kernelSumDensity
	( double *share, cdouble *src, cint no, cint tx, cint ty, cint tz );

extern __global__ void kernelGradient
	( double *div, double *p, cdouble *vel_u, cdouble *vel_v, cdouble *vel_w, cint tx, cint ty, cint tz );

extern __global__ void kernelSubtract
	( double *vel_u, double *vel_v, double *vel_w, cdouble *p, cint tx, cint ty, cint tz );

extern __global__ void kernelAddSource
	( double *density, double *vel_u, double *vel_v, double *vel_w, cint tx, cint ty, cint tz );

extern __global__ void kernelPickData
	( uchar *c, cdouble *bufs, int ofi, int ofj, int ofk, cint tx, cint ty, cint tz );

extern __global__ void kernelInterRootGrids
	( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate, cint tx, cint ty, cint tz );

extern __global__ void kernelInterLeafGrids
	( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate, cint tx, cint ty, cint tz );

extern __global__ void kernelClearHalo( double *grids, cint tx, cint ty, cint tz );

extern __global__ void kernelHandleHalo
	( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back,
	cint tx, cint ty, cint tz );

#endif