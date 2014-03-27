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
#include "FrameworkDynamic.h"
#include "ISO646.h"

using std::vector;
using std::string;

namespace sge
{
	struct SimNode
	{
		SGBOOLEAN updated;
		SGBOOLEAN active;
		int x, y, z;
		SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};
	
	class FluidSimProc
	{
	private:
		vector <double*> m_vectBulletBufs;
		vector <double*> m_vectCompBufs;

		vector<double*> m_vectToyDens, m_vectHostDens;
		vector<double*> m_vectToyVelU, m_vectHostVelU;
		vector<double*> m_vectToyVelV, m_vectHostVelV;
		vector<double*> m_vectToyVelW, m_vectHostVelW;
		vector<double*> m_vectToyObst, m_vectHostObst;

		SGUCHAR *m_ptrDeviceVisual, *m_ptrHostVisual;
		double *m_ptrHostToy, *m_ptrHostComp, *m_ptrHostBullet;
		
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

        void SaveCurStage( void );

        void LoadPreStage( void );

	private:
		int ix(cint i, cint j, cint k, cint tiles ) { return k * tiles * tiles + j * tiles + i; };

		int ix(cint i, cint j, cint k, cint tilex, cint tiley) { return k * tilex * tiley + j * tilex + i; };

		void AssembleBuffers( void );

		void DeassembleBuffers( void );

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

#define __device_func__ <<<gridDim,blockDim>>>

#define BUL_BUFS             11
#define dev_den              m_vectBulletBufs[ 0 ]
#define dev_den0             m_vectBulletBufs[ 1 ]
#define dev_u                m_vectBulletBufs[ 2 ]
#define dev_u0               m_vectBulletBufs[ 3 ]
#define dev_v                m_vectBulletBufs[ 4 ]
#define dev_v0               m_vectBulletBufs[ 5 ]
#define dev_w                m_vectBulletBufs[ 6 ]
#define dev_w0               m_vectBulletBufs[ 7 ]
#define dev_div              m_vectBulletBufs[ 8 ]
#define dev_p                m_vectBulletBufs[ 9 ]
#define dev_obs              m_vectBulletBufs[ 10 ]

#define COMP_BUFS            5
#define comp_den             m_vectCompBufs[0]
#define comp_u               m_vectCompBufs[1]
#define comp_v               m_vectCompBufs[2]
#define comp_w               m_vectCompBufs[3]
#define comp_obst            m_vectCompBufs[4]

#endif