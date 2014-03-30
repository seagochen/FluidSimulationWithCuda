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

#define gGrids_density       m_vectgGrids[0]
#define gGrids_velocity_u    m_vectgGrids[1]
#define gGrids_velocity_v    m_vectgGrids[2]
#define gGrids_velocity_w    m_vectgGrids[3]
#define gGrids_obstacle      m_vectgGrids[4]

#define gBullets_density     m_vectgBullets[0]
#define gBullets_velocity_u  m_vectgBullets[1]
#define gBullets_velocity_v  m_vectgBullets[2]
#define gBullets_velocity_w  m_vectgBullets[3]
#define gBullets_obstacle    m_vectgBullets[4]
#define gBullets_density0    m_vectgBullets[5]
#define gBullets_velocity_u0 m_vectgBullets[6]
#define gBullets_velocity_v0 m_vectgBullets[7]
#define gBullets_velocity_w0 m_vectgBullets[8]
#define gBullets_divergence  m_vectgBullets[9]
#define gBullets_pressure    m_vectgBullets[10]
	
#define Big_density          m_vectBigBuffers[0]
#define Big_velocity_u       m_vectBigBuffers[1]
#define Big_velocity_v       m_vectBigBuffers[2]
#define Big_velocity_w       m_vectBigBuffers[3]
#define Big_obstacle         m_vectBigBuffers[4]

#define sGrids_density       m_vectsGrids[0]
#define sGrids_velocity_u    m_vectsGrids[1]
#define sGrids_velocity_v    m_vectsGrids[2]
#define sGrids_velocity_w    m_vectsGrids[3]
#define sGrids_obstacle      m_vectsGrids[4]

#define sBullets_density     m_vectsBullets[0]
#define sBullets_velocity_u  m_vectsBullets[1]
#define sBullets_velocity_v  m_vectsBullets[2]
#define sBullets_velocity_w  m_vectsBullets[3]
#define sBullets_obstacle    m_vectsBullets[4]
#define sBullets_density0    m_vectsBullets[5]
#define sBullets_velocity_u0 m_vectsBullets[6]
#define sBullets_velocity_v0 m_vectsBullets[7]
#define sBullets_velocity_w0 m_vectsBullets[8]
#define sBullets_divergence  m_vectsBullets[9]
#define sBullets_pressure    m_vectsBullets[10]

	class FluidSimProc
	{
	private:
		pdouble dev_den, dev_u, dev_v, dev_w, dev_p, dev_div, dev_obs, dev_den0, dev_u0, dev_v0, dev_w0;

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

	private:
		void SolveNavierStokesEquation( cdouble dt, bool add, bool vel, bool dens, cint bx, cint by, cint bz );

		void DensitySolver( cdouble dt, cint bx, cint by, cint bz );

		void VelocitySolver( cdouble dt, cint bx, cint by, cint bz );

		void SourceSolver( cdouble dt, cint bx, cint by, cint bz );

		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor, cint bx, cint by, cint bz );

		void Advection( double *out, cdouble *in, cdouble *u, cdouble *v, cdouble *w, cdouble dt, cint bx, cint by, cint bz );

		void Diffusion( double *out, cdouble *in, cdouble diff, cint bx, cint by, cint bz );

		void Projection( double *u, double *v, double *w, double *div, double *p, cint bx, cint by, cint bz );
	};
};

#endif