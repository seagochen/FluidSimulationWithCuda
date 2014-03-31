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
#include "FunctionHelper.h"
#include "Framework.h"
#include "ISO646.h"

using std::vector;
using std::string;

namespace sge
{	
#define __device_func__ <<<gridDim,blockDim>>>

#define DEV_DENSITY     0
#define DEV_VELOCITY_U  1
#define DEV_VELOCITY_V  2
#define DEV_VELOCITY_W  3
#define DEV_OBSTACLE    4
#define DEV_DIVERGENCE  5
#define DEV_PRESSURE    6
#define DEV_DENSITY0    7
#define DEV_VELOCITY_U0 8
#define DEV_VELOCITY_V0 9
#define DEV_VELOCITY_W0 10

	class FluidSimProc
	{
	private:
		double **dev_den, **dev_u, **dev_v, **dev_w, **dev_p, **dev_div, **dev_obs,
			**dev_den0, **dev_u0, **dev_v0, **dev_w0;

		vector<double*> m_vectgGrids, m_vectgBullets;
		vector<double*> m_vectsGrids, m_vectsBullets;
		vector<double*> m_vectBigBuffers;

		SGUCHAR *m_ptrDeviceVisual, *m_ptrHostVisual;
				
		dim3 gridDim, blockDim;

		string m_szTitle;

	private:
		FunctionHelper m_scHelper;

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
		int ix(cint i, cint j, cint k, cint tiles ) { return k * tiles * tiles + j * tiles + i; };

		int ix(cint i, cint j, cint k, cint tilex, cint tiley) { return k * tilex * tiley + j * tilex + i; };

		void GenerVolumeImg( void );

		void SolveGlobalFlux( void );

		void RefinementFlux( void );

	private:
		void SolveNavierStokesEquation
			( cdouble dt, bool add, bool vel, bool dens,
			cint tx, cint ty,
			cint gx, cint gy, cint gz,
			cint bx, cint by, cint bz );

		void DensitySolver( cdouble dt, 
			cint bx, cint by, cint bz );

		void VelocitySolver( cdouble dt,
			cint bx, cint by, cint bz );

		void SourceSolver( cdouble dt,
			cint bx, cint by, cint bz );

		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor,
			cint bx, cint by, cint bz );

		void Advection( double *out, cdouble *in, cdouble *u, cdouble *v, cdouble *w, cdouble dt,
			cint bx, cint by, cint bz );

		void Diffusion( double *out, cdouble *in, cdouble diff,
			cint bx, cint by, cint bz );

		void Projection( double *u, double *v, double *w, double *div, double *p,
			cint bx, cint by, cint bz );
	};
};

#endif