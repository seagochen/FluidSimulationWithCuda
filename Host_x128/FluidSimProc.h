/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 24, 2014
* <File Name>     FluidSimProc.h
*/


#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "FrameworkDynamic.h"
#include "ISO646.h"

using std::vector;
using std::string;

namespace sge
{	
	class FluidSimProc
	{
	private:
		double *dev_u, *dev_v, *dev_w, *dev_u0, *dev_v0, *dev_w0;
		double *dev_den, *dev_den0, *dev_p, *dev_obs;

		SGUCHAR *m_visual;			

		string m_szTitle;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

	public:
		void ClearBuffers( void );

		sstr GetTitleBar( void ) { return &m_szTitle; };

		void FreeResource( void );

		void AllocateResource( void );

		void InitParams( FLUIDSPARAM *fluid );

		void RefreshStatus( FLUIDSPARAM *fluid );

		void FluidSimSolver( FLUIDSPARAM *fluid );

		void InitBoundary( void );

	private:
		int ix(cint i, cint j, cint k ) { return k * 128 * 128 + j * 128 + i; };

		void GenerVolumeImg( void );

	private:
		void SolveNavierStokesEquation( cdouble dt, bool add, bool vel, bool dens );

		void SolveGlobal( cdouble dt, bool add, bool vel, bool dens );

		void DensitySolverGlobal( cdouble dt );

		void VelocitySolverGlobal( cdouble dt );

		void SourceSolverGlobal( cdouble dt );

		void JacobiGlobal( double *out, cdouble *in, cdouble diff, cdouble divisor );

		void AdvectionGlobal( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w );

		void DiffusionGlobal( double *out, cdouble *in, cdouble diff );

		void ProjectionGlobal( double *u, double *v, double *w, double *div, double *p );
	};
};

#endif