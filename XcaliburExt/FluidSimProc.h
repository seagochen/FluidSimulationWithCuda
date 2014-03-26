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

	enum SGFILEDTYPE
	{
		SG_DENSITY,
		SG_VELOCITY_U,
		SG_VELOCITY_V,
		SG_VELOCITY_W,
		SG_OBSTACLE,
		SG_DIV,
		SG_PRESSURE,
	};
	

#define GLOBAL_BUFS    4
#define global_dens    m_vectDevGlobal[0]
#define global_velu    m_vectDevGlobal[1]
#define global_velv    m_vectDevGlobal[2]
#define global_velw    m_vectDevGlobal[3]

#define TEMP_BUFS      4
#define temp_u         m_vectGPUTempBufs[0]
#define temp_v         m_vectGPUTempBufs[1]
#define temp_w         m_vectGPUTempBufs[2]
#define temp_d         m_vectGPUTempBufs[3]

	class FluidSimProc
	{
	private:
		vector<double*> m_vectHostDens, m_vectHostObst,
			m_vectHostVelU, m_vectHostVelV, m_vectHostVelW;

		vector<double*> m_vectDevDens, m_vectDevObst, m_vectDevGlobal,
			m_vectDevVelU, m_vectDevVelV, m_vectDevVelW;

		vector<double*> m_vectGPUDens, m_vectGPUDiv, m_vectGPUPress, m_vectGPUTempBufs,
			m_vectGPUVelU, m_vectGPUVelV, m_vectGPUVelW, m_vectGPUObst;

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

	private:
		void SolveNavierStokesEquation( cdouble dt, bool add, bool vel, bool dens );

		void DensitySolver( cdouble dt );

		void VelocitySolver( cdouble dt );

		void SourceSolver( cdouble dt );
	};
};


#define __device_func__ <<<gridDim,blockDim>>>

#define GridsParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z )

#define BulletParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 22, 22, BULLET_X, BULLET_Y, BULLET_Z )

#define _zero(vect) kernelZeroBuffers __device_func__ ( vect, GRIDS_X, GRIDS_Y, GRIDS_Z )


#endif