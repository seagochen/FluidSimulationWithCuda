/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Apr 03, 2014
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
		double *u, *v, *w, *u0, *v0, *w0;
		double *den, *den0, *p, *obs, *div;

		SGUCHAR *visual;			

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
		inline int ix(cint i, cint j, cint k ) { return k * 128 * 128 + j * 128 + i; };

		void GenerVolumeImg( void );

	private:
		void SolveNavierStokesEquation
			( cdouble dt, bool add, bool vel, bool dens );

		void DensitySolver( cdouble dt );

		void VelocitySolver( cdouble dt );

		void SourceSolver( cdouble dt );

		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor );

		void Advection( double *out, cdouble *in, cdouble *u, cdouble *v, cdouble *w, cdouble dt );

		void Diffusion( double *out, cdouble *in, cdouble diff );

		void Projection( double *u, double *v, double *w, double *div, double *p );
	};
};

#endif