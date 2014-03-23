/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 23, 2014
* <File Name>     NavierStokesSolver.cu
*/

#include <time.h>
#include <iostream>
#include <utility>
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "FrameworkDynamic.h"
#include "Kernels.h"
#include "SimulationParam.h"

using namespace sge;

#define __device_func__ <<<gridDim, blockDim>>>

/************************************************************************************
** The following implement functions are member of class NavierStokesSolver        **
*************************************************************************************/

void NavierStokesSolver::SolveSource( double *ptrDevDens, double *ptrDevU, double *ptrDevV, double *ptrDevW,
			cdouble *ptrDevObst, int *nInTime, int *nDeTime, cdouble deltatime )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( *nDeTime eqt 0 )
	{
		kernelAddSource __device_func__ ( ptrDevDens, ptrDevU, ptrDevV, ptrDevW, 
			ptrDevObst, DENSITY, VELOCITY, deltatime, time(NULL), BULLET_X, BULLET_Y, BULLET_Z );

		*nInTime++;

		if ( *nInTime eqt 200 )
		{
			*nDeTime = *nInTime;
			*nInTime = 0;
		}
	}
	else
	{
		*nDeTime--;
	}
};

void NavierStokesSolver::SolveDensity( double *ptrDevDens0, double *ptrDevDens,
									   cdouble *ptrDevU, cdouble *ptrDevV, cdouble *ptrDevW,
									   cdouble timestep )
{
	Diffusion( ptrDevDens0, ptrDevDens, DIFFUSION );
	std::swap( ptrDevDens0, ptrDevDens );
	Advection ( ptrDevDens0, ptrDevDens, timestep, ptrDevU, ptrDevV, ptrDevW );
	std::swap( ptrDevDens0, ptrDevDens );
};

void NavierStokesSolver::SolveVelocity(  double *ptrDevU0, double *ptrDevV0, double *ptrDevW0,
									   double *ptrDevU, double *ptrDevV, double *ptrDevW,
									   double *ptrDiv,  double *ptrPres, cdouble timestep )
{
	// diffuse the velocity field (per axis):
	Diffusion( ptrDevU0, ptrDevU, VISOCITY );
	Diffusion( ptrDevV0, ptrDevV, VISOCITY );
	Diffusion( ptrDevW0, ptrDevW, VISOCITY );

	std::swap( ptrDevU0, ptrDevU );
	std::swap( ptrDevV0, ptrDevV );
	std::swap( ptrDevW0, ptrDevW );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( ptrDevU, ptrDevV, ptrDevW, ptrDiv, ptrPres );
	
	// advect the velocity field (per axis):
	Advection( ptrDevU0, ptrDevU, timestep, ptrDevU, ptrDevV, ptrDevW );
	Advection( ptrDevV0, ptrDevV, timestep, ptrDevU, ptrDevV, ptrDevW );
	Advection( ptrDevW0, ptrDevW, timestep, ptrDevU, ptrDevV, ptrDevW );

	std::swap( ptrDevU0, ptrDevU );
	std::swap( ptrDevV0, ptrDevV );
	std::swap( ptrDevW0, ptrDevW );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( ptrDevU, ptrDevV, ptrDevW, ptrDiv, ptrPres );
};

void NavierStokesSolver::Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelAdvection __device_func__ ( out, in, timestep, u, v, w, BULLET_X, BULLET_Y, BULLET_Z );
};

void NavierStokesSolver::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
	
	for ( int k = 0; k < 20; k++ )
		kernelJacobi __device_func__ ( out, in, diff, divisor, BULLET_X, BULLET_Y, BULLET_Z );	
};

void NavierStokesSolver::Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_X * GRIDS_X;

	Jacobi( out, in, rate, 1+6*rate );
};

void NavierStokesSolver::Projection( double *u, double *v, double *w, double *div, double *p )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// the velocity gradient
	kernelGradient __device_func__ ( div, p, u, v, w, BULLET_X, BULLET_Y, BULLET_Z );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract __device_func__ ( u, v, w, p, BULLET_X, BULLET_Y, BULLET_Z );
};