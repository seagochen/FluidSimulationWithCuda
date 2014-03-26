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
		SGBOOLEAN updated;
		SGBOOLEAN active;
		int x, y, z;
		SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};
	
	class FluidSimProc
	{
	private:
		vector <double*> m_vectCompBufs;

		vector<double*> m_vectHostDens, m_vectGPUDens, m_vectNewDens;
		vector<double*> m_vectHostVelU, m_vectGPUVelU, m_vectNewVelU;
		vector<double*> m_vectHostVelV, m_vectGPUVelV, m_vectNewVelV;
		vector<double*> m_vectHostVelW, m_vectGPUVelW, m_vectNewVelW;
		vector<double*> m_vectHostObst, m_vectGPUObst;

		vector<double*> m_vectGlobalBufs;

		SGUCHAR *m_ptrDeviceVisual, *m_ptrHostVisual;
		
		dim3 gridDim, blockDim;

		string m_szTitle;

	private:
		FunctionHelper m_scHelper;


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

	private:
		void SolveNavierStokesEquation( cdouble timestep, bool add );

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

#define Dim3ParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z )

#define _zero(vect) kernelZeroBuffers __device_func__ ( vect, GRIDS_X, GRIDS_Y, GRIDS_Z )

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

#define dens_C               m_vectCompBufs[ 0 ]
#define dens_L               m_vectCompBufs[ 11 ]
#define dens_R               m_vectCompBufs[ 12 ]
#define dens_U               m_vectCompBufs[ 13 ]
#define dens_D               m_vectCompBufs[ 14 ]
#define dens_F               m_vectCompBufs[ 15 ]
#define dens_B               m_vectCompBufs[ 16 ]

#define velu_C               m_vectCompBufs[ 2 ]
#define velu_L               m_vectCompBufs[ 17 ] 
#define velu_R               m_vectCompBufs[ 18 ]
#define velu_U               m_vectCompBufs[ 19 ]
#define velu_D               m_vectCompBufs[ 20 ]
#define velu_F               m_vectCompBufs[ 21 ]
#define velu_B               m_vectCompBufs[ 22 ]

#define velv_C               m_vectCompBufs[ 4 ]
#define velv_L               m_vectCompBufs[ 23 ]
#define velv_R               m_vectCompBufs[ 24 ]
#define velv_U               m_vectCompBufs[ 25 ]
#define velv_D               m_vectCompBufs[ 26 ]
#define velv_F               m_vectCompBufs[ 27 ]
#define velv_B               m_vectCompBufs[ 28 ]

#define velw_C               m_vectCompBufs[ 6 ]
#define velw_L               m_vectCompBufs[ 29 ]
#define velw_R               m_vectCompBufs[ 30 ]
#define velw_U               m_vectCompBufs[ 31 ]
#define velw_D               m_vectCompBufs[ 32 ]
#define velw_F               m_vectCompBufs[ 33 ]
#define velw_B               m_vectCompBufs[ 34 ]

#define temp_u               m_vectNewTemp[0]
#define temp_v               m_vectNewTemp[1]
#define temp_w               m_vectNewTemp[2]

#endif