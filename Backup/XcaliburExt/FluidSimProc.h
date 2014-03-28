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
	

#define COMP_BUFS      5
#define comp_dens      m_vectCompBufs[0]
#define comp_velu      m_vectCompBufs[1]
#define comp_velv      m_vectCompBufs[2]
#define comp_velw      m_vectCompBufs[3]
#define comp_obst      m_vectCompBufs[4]

#define BULLET_BUFS    11
#define dev_u          m_vectBulletBufs[0]
#define dev_v          m_vectBulletBufs[1]
#define dev_w          m_vectBulletBufs[2]
#define dev_obs        m_vectBulletBufs[3]
#define dev_den        m_vectBulletBufs[4]
#define dev_div        m_vectBulletBufs[5]
#define dev_prs        m_vectBulletBufs[6]
#define dev_u0         m_vectBulletBufs[7]
#define dev_v0         m_vectBulletBufs[8]
#define dev_w0         m_vectBulletBufs[9]
#define dev_den0       m_vectBulletBufs[10]

	class FluidSimProc
	{
	private:
		vector<double*> m_vectHostDens, m_vectHostObst,
			m_vectHostVelU, m_vectHostVelV, m_vectHostVelW;

		vector<double*> m_vectDevDens, m_vectDevObst, 
			m_vectDevVelU, m_vectDevVelV, m_vectDevVelW;

		vector<double*> m_vectCompBufs;

		vector<double*> m_vectBulletBufs;

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

#define CompParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, COMPS_X, COMPS_Y, COMPS_Z )

#define BulletParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 26, 26, COMPS_X, COMPS_Y, COMPS_Z )

#define VisualParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 128, 8, COMPS_X, COMPS_Y, COMPS_Z )

#endif