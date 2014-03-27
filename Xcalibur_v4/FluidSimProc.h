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
		/* temporary buffers for fluid simulation */
		vector <double*> m_vectCompBufs;

		vector<double*> m_vectHostDens, m_vectGPUDens, m_vectNewDens;
		vector<double*> m_vectHostVelU, m_vectGPUVelU, m_vectNewVelU;
		vector<double*> m_vectHostVelV, m_vectGPUVelV, m_vectNewVelV;
		vector<double*> m_vectHostVelW, m_vectGPUVelW, m_vectNewVelW;
		vector<double*> m_vectHostObst, m_vectGPUObst, m_vectNewObst;

		/* visualization */
		SGUCHAR *m_ptrDeviceVisual, *m_ptrHostVisual;
		
		/* CUDA */
		dim3 gridDim, blockDim;

		/* title bar */
		string m_szTitle;

	private:
		FunctionHelper m_scHelper;


	public:
		FluidSimProc( FLUIDSPARAM *fluid );

	public:
		/* clear device buffers */
		void ClearBuffers( void );

		/* get the title of windows's bar */
		sstr GetTitleBar( void ) { return &m_szTitle; };

		void FreeResource( void );

		void AllocateResource( void );

		void InitParams( FLUIDSPARAM *fluid );

		void RefreshStatus( FLUIDSPARAM *fluid );

		void FluidSimSolver( FLUIDSPARAM *fluid );

	private:
		void SolveNavierStokesEquation( cdouble timestep, bool add );
		void DensitySolver( cdouble timestep );
		void VelocitySolver( cdouble timestep );
		void SourceSolver( void );
		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor );
		void Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w );
		void Diffusion( double *out, cdouble *in, cdouble diff );
		void Projection( double *u, double *v, double *w, double *div, double *p );
	};
};

#define __device_func__ <<<gridDim,blockDim>>>
#define DeviceParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _zero(vect) kernelZeroBuffers __device_func__ ( vect, GRIDS_X, GRIDS_Y, GRIDS_Z )

#define COMP_BUFS            11
#define dev_den              m_vectCompBufs[ 0 ]
#define dev_den0             m_vectCompBufs[ 1 ]
#define dev_u                m_vectCompBufs[ 2 ]
#define dev_u0               m_vectCompBufs[ 3 ]
#define dev_v                m_vectCompBufs[ 4 ]
#define dev_v0               m_vectCompBufs[ 5 ]
#define dev_w                m_vectCompBufs[ 6 ]
#define dev_w0               m_vectCompBufs[ 7 ]
#define dev_div              m_vectCompBufs[ 8 ]
#define dev_p                m_vectCompBufs[ 9 ]
#define dev_obs              m_vectCompBufs[ 10 ]

#endif