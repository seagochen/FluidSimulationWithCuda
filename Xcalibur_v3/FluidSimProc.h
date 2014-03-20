/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 19, 2014
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

using std::vector;
using std::string;

namespace sge
{
	struct SimNode
	{
		bool updated;
		bool active;
		int x, y, z;
		SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};

	class FluidSimProc
	{
	private:
		/* temporary buffers for fluid simulation */
		vector <double*> m_vectGPUBuffers;

		/* nodes for gpu and host */
		vector <double*> m_vectGPUDens, m_vectHostDens, m_vectNewDens;
		vector <double*> m_vectGPUVelU, m_vectHostVelU, m_vectNewVelU;
		vector <double*> m_vectGPUVelV, m_vectHostVelV, m_vectNewVelV;
		vector <double*> m_vectGPUVelW, m_vectHostVelW, m_vectNewVelW;
		vector <double*> m_vectGPUObst, m_vectHostObst;

		/* topology of nodes on host and device */
		vector <SimNode*> m_vectLink;

		/* visualization buffers */
		uchar *m_ptrGPUVisual, *m_ptrHostVisual;

		/* temporary buffers for some purpose */
		double *m_ptrGPUShare, *m_ptrHostShare;

		/* node and volumetric size */
		size_t m_nNodeSize, m_nVolumSize, m_nBulletSize;
		size_t m_nNodeNum, m_nBulletNum;

		/* etc. */
		int m_nDensIncrease, m_nDensDecrease;
		
		FunctionHelper m_scHelper;
		SimNode *m_ptrSimNode;
		string m_szTitle;	

		dim3 gridDim, blockDim;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

	private:
		int IX( cint i, cint j, cint k, cint ts ) { return k * ts * ts + j * ts + i; };
		int IX( cint i, cint j, cint k, cint tx, cint ty, cint tz ) { return k * tx * ty + j * tx + i; };

	public:
		void InitParams( FLUIDSPARAM *fluid );
		void AllocateResource( void );
		void CreateTopology( void );
		sstr GetTitleBar( void ) { return &m_szTitle; };
		void ZeroBuffers( void );
		void InitBoundary( void ); // TODO: ªπŒ¥ µœ÷
		void FreeResource( void );
		void FluidSimSolver( FLUIDSPARAM *fluid );

	private:
		bool CreateCompNodes( void );
		bool CreateDeviceBuffers( void );
		bool CreateVisualBuffers( void );

		void ClearCompNodes( void );
		void ClearDeviceBuffers( void );
		void ClearVisualBuffers( void );

		void ReleaseCompNodes( void );
		void ReleaseDeviceBuffers( void );
		void ReleaseVisualBuffers( void );


	/****************************************************************************************/
		
//	private:
//		void LoadBullet( int i, int j, int k );
//		void ExitBullet( int i, int j, int k );
//		void clearBullet( void );
//		void pickNodeToBullet( int i, int j, int k );
//		void pickNeighborsToBullet( int i, int j, int k );
//		void pickBulletToNode( int i, int j, int k );
//		void pickImgFromNode( int i, int j, int k );
//
//	private:
//		
//		void RefreshStatus( FLUIDSPARAM *fluid );
//
//	public:
//		
//		void SolveRootNode( void );
//		void SolveLeafNode( void );
//
//	private:
//		void SolveNavierStokesEquation( cdouble timestep, bool add );
//		void DensitySolver( cdouble timestep );
//		void VelocitySolver( cdouble timestep );
//		void AddSource( void );
//		
//		void ReadBuffers( void );
//		void WriteBuffers( void );
//		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor );
//		void Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w );
//		void Diffusion( double *out, cdouble *in, cdouble diff );
//		void Projection( double *u, double *v, double *w, double *div, double *p );
	};
};

#endif