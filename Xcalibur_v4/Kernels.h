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

__global__ void kernelSumDensity( double *share, cdouble *src, cint no );

#endif