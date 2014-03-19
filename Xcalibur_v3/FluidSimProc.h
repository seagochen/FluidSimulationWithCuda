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
		vector <double*> m_vectGPUDens, m_vectHostDens;
		vector <double*> m_vectGPUVelU, m_vectHostVelU;
		vector <double*> m_vectGPUVelV, m_vectHostVelV;
		vector <double*> m_vectGPUVelW, m_vectHostVelW;
		vector <int*> m_vectGPUObst, m_vectHostObst;

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

	public:
		void AllocateResource( void );

	private:
		bool CreateCompNodesResource( void );
		bool CreateBulletResource( void );
		bool CreateResource( void );


	public:
		void ZeroBuffers( void );
		sstr GetTitleBar( void ) { return &m_szTitle; };
		void PrintMSG( void );
		void HostToDevice( void );
		void DeviceToHost( void );
		void FreeResource( void );

	private:
		void zeroDeivceRes( void );
		void zeroHostRes( void );
		void zeroVisualBuffers( void );
		void zeroShareBuffers( void );
		void zeroTempoBuffers( void );
		void zeroGlobalNode( void );

	private:
		void freeHostRes( void );
		void freeDeviceRes( void );
		void freeShareBuffers( void );
		void freeVisualBuffers( void );

	private:
		void LoadBullet( int i, int j, int k );
		void ExitBullet( int i, int j, int k );
		void clearBullet( void );
		void pickNodeToBullet( int i, int j, int k );
		void pickNeighborsToBullet( int i, int j, int k );
		void pickBulletToNode( int i, int j, int k );
		void pickImgFromNode( int i, int j, int k );

	private:
		void InitParams( FLUIDSPARAM *fluid );
		void CreateTopology( void );
		void RefreshStatus( FLUIDSPARAM *fluid );

	public:
		void FluidSimSolver( FLUIDSPARAM *fluid );
		void SolveRootNode( void );
		void SolveLeafNode( void );

	private:
		void SolveNavierStokesEquation( cdouble timestep, bool add );
		void DensitySolver( cdouble timestep );
		void VelocitySolver( cdouble timestep );
		void AddSource( void );
		void InitBoundary( void );
		void ReadBuffers( void );
		void WriteBuffers( void );
		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor );
		void Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w );
		void Diffusion( double *out, cdouble *in, cdouble diff );
		void Projection( double *u, double *v, double *w, double *div, double *p );
	};
};

#endif