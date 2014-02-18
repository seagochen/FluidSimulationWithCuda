/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 02, 2014
* <Last Time>     Feb 02, 2014
* <File Name>     CUDAInterfaces.h
*/

#ifndef __cuda_interfaces_h_
#define __cuda_interfaces_h_

#include "DataStructures.h"

#define VELOCITY_FIELD_U    0
#define VELOCITY_FIELD_V    1
#define VELOCITY_FIELD_W    2
#define DENSITY_FIELD       3

using namespace sge;

/* 采集网格数据，并转换为volumetric data */
extern void hostPickData( SGUCHAR *data, const double *bufs, SGINT3 *nodeIX );

/* clear GPU buffer */
extern void hostZeroBuffer( double *grid );

extern void DensitySolver
	( double *dev_den, double *dev_u, double *dev_v, double *dev_w, double *dev_den0 );

extern void VelocitySolver
	( double *dev_u, double *dev_v, double *dev_w, double *dev_u0, double *dev_v0, double *dev_w0,
	double *div, double *p );

#endif