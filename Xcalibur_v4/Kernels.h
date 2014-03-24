/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 04, 2014
* <File Name>     Kernels.h
*/

#ifndef __kernels_h__
#define __kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"

extern __global__ void kernelJacobi
	( double *out, cdouble *in, cdouble diffusion, cdouble divisor );


extern __global__ void kernelAdvection
	( double *out, cdouble *in, cdouble delta,
	cdouble *u, cdouble *v, cdouble *w );

extern __global__ void kernelGradient( double *div, double *prs,
							   cdouble *u, cdouble *v, cdouble *w );

extern __global__ void kernelSubtract( double *u, double *v, double *w, double *prs );

extern __global__ void kernelZeroGrids( double *grid );

extern __global__ void kernelZeroVolumetric( uchar *visual );

extern __global__ void kernelZeroShareBuffers( double *bufs );

extern __global__ void kernelZeroShareBuffers( int *bufs );;

extern __global__ void kernelCopyGrids( double *src, cdouble *dst );

extern __global__ void kernelObstacle( double *grids, cdouble *obstacle, cint field );

extern __global__ void kernelAddSource( double *density, double *vel_u, double *vel_v, double *vel_w );

extern __global__ void kernelPickData( uchar *c, cdouble *bufs, int ofi, int ofj, int ofk, cint grids );

extern __global__ void kernelInterRootGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate );

extern __global__ void kernelInterLeafGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate );

extern __global__ void kernelClearHalo( double *grids );

extern __global__ void kernelHandleHalo
	( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back );

extern __global__ void kernelSumDensity( double *share, cdouble *src, cint no );

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