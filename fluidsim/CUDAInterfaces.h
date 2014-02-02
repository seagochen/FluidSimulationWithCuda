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

extern void VelocitySolver
	( double *u, double *v, double *w, double *div, double *p,
	double *u0, double *v0, double *w0, sge::SGDEVICEBUFF *global, double *stores );

extern void DensitySolver
	( double *dens, double *dens0, sge::SGDEVICEBUFF *global, double *stores );

extern void AddSource
	( double *buffer, sge::SGSTDGRID *grids, sge::SGFIELDTYPE type );

extern void HaloDataExchange
	( double *buffer, sge::SGDEVICEBUFF *global, sge::SGFIELDTYPE type, sge::SGNODECOORD coord );

#endif