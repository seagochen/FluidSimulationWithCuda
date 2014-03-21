/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 20, 2014
* <File Name>     Navier_Stokes_Kernels.h
*/

#ifndef __navier_stokes_kernels_h__
#define __navier_stokes_kernels_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"

extern __global__ void kernelJacobi( double *out, cdouble *in, cdouble diffusion, cdouble divisor,
									cint tx, cint ty, cint tz );

extern __global__ void kernelAdvection( double *out, cdouble *in, cdouble delta,
									   cdouble *u, cdouble *v, cdouble *w,
									   cint tx, cint ty, cint tz );

extern __global__ void kernelGradient( double *div, double *prs,
							cdouble *u, cdouble *v, cdouble *w,
							cint tx, cint ty, cint tz );

extern __global__ void kernelSubtract( double *u, double *v, double *w, double *prs, 
							  cint tx, cint ty, cint tz );

extern __global__ void kernelAddSource( double *den, double *u, double *v, double *w, 
									   cdouble *obst, cdouble rho, cdouble vel, cdouble delta, cint time,
									   cint tx, cint ty, cint tz );

extern __global__ void kernelDensToVolume( uchar *volume, cdouble *rho,
								   cdouble scale, int offi, int offj, int offk,
								   cint tx, cint ty, cint tz, cint vx, cint vy, cint vz );

extern __global__ void kernelDataFromRoot( double *dst, cdouble *src, 
								   cdouble offx, cdouble offy, cdouble offz, cdouble scale,
								   cint tx, cint ty, cint tz );

extern __global__ void kernelDataFromLeaf( double *dst, cdouble *src, 
								   cdouble offx, cdouble offy, cdouble offz, cdouble scale,
								   cint tx, cint ty, cint tz );

extern __global__ void kernelHandleHalo( double *bullet, 
								 cdouble *left,  cdouble *right,
								 cdouble *up,    cdouble *down,
								 cdouble *front, cdouble *back,
								 cint tx, cint ty, cint tz, cint bx, cint by, cint bz );

#endif