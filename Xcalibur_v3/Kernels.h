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


#include "Navier_Stokes_Kernels.h"

#endif