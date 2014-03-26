/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 26, 2014
* <File Name>     FluidSimProc.h
*/


#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "FunctionHelper.h"
#include "FrameworkDynamic.h"
#include "ISO646.h"

using std::vector;
using std::string;

namespace sge
{
	struct SimNode
	{
		int updated, active;
		int x, y, z;
		SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};

	enum SGHIERARCHY
	{
		SG_LARGE_SCALE,
		SG_HIGH_PRECISION,
	};
	
	class FluidSimProc
	{
	private:
		vector <double*> m_vectCompBufs, m_vectGlobalBufs;

		vector<double*> m_vectHostDens, m_vectGPUDens, m_vectNewDens;
		vector<double*> m_vectHostVelU, m_vectGPUVelU, m_vectNewVelU;
		vector<double*> m_vectHostVelV, m_vectGPUVelV, m_vectNewVelV;
		vector<double*> m_vectHostVelW, m_vectGPUVelW, m_vectNewVelW;
		vector<double*> m_vectHostObst, m_vectGPUObst;
		vector<SimNode*> m_link;

		SGUCHAR *m_ptrDeviceVisual, *m_ptrHostVisual;

	private:
		FunctionHelper m_scHelper;
		SGHIERARCHY m_enHierarchy;
		string m_szTitle;
		dim3 gridDim, blockDim;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

	public:
		inline int ix(cint i, cint j, cint k, cint tiles ) { return k * tiles * tiles + j * tiles + i; };

		inline int ix(cint i, cint j, cint k, cint tilex, cint tiley)
		{ return k * tilex * tiley + j * tilex + i; };

	public:
		void ClearBuffers( void );

		sstr GetTitleBar( void ) { return &m_szTitle; };

		void FreeResource( void );

		void AllocateResource( void );

		void InitParams( FLUIDSPARAM *fluid );

		void InitBound( void );

		void RefreshStatus( FLUIDSPARAM *fluid );

		void FluidSimSolver( FLUIDSPARAM *fluid );

		void SetHierarchy( SGHIERARCHY hierarchy ) { m_enHierarchy = hierarchy; };

		void LoadPreStage( void );

		void SaveCurStage( void );

		void SetCurrentNode( int i, int j, int k );

		void GetCurrentNode( int i, int j, int k );

	private:
		void SolveNavierStokesEquation( cdouble dt, bool add, bool dens, bool vel );

		void DensitySolver( cdouble timestep );

		void VelocitySolver( cdouble timestep );

		void SourceSolver( cdouble timestep );

		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor );

		void Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w );

		void Diffusion( double *out, cdouble *in, cdouble diff );

		void Projection( double *u, double *v, double *w, double *div, double *p );
	};
};


#define __device_func__ <<<gridDim,blockDim>>>

#define GridsParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z )

#define BulletParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 22, 22, BULLET_X, BULLET_Y, BULLET_Z )

#define _zero(vect) kernelZeroBuffers __device_func__ ( vect, GRIDS_X, GRIDS_Y, GRIDS_Z )


#endif