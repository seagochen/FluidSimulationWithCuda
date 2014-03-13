/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 07, 2014
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
	class FluidSimProc
	{

#pragma region inner structures
	private:
		struct SimNode
		{
			SGBOOLEAN updated;
			SGBOOLEAN active;
			int x, y, z;
			SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		};
#pragma endregion

#pragma region private variables
	private:
		/* temporary buffers for fluid simulation */
		vector <double*> dev_buffers;

		/* nodes for gpu and host */
		vector <double*> dev_density,    host_density;
		vector <double*> dev_velocity_u, host_velocity_u;
		vector <double*> dev_velocity_v, host_velocity_v;
		vector <double*> dev_velocity_w, host_velocity_w;
		vector <double*> dev_obstacle,   host_obstacle;

		/* local nodes */
		vector <double*> node_density;
		vector <double*> node_velocity_u;
		vector <double*> node_velocity_v;
		vector <double*> node_velocity_w;
		vector <double*> node_obstacle;

		double *gd_density, *gd_velocity_u, *gd_velocity_v, *gd_velocity_w, *gd_obstacle;

		/* topology of nodes on host and device */
		vector <SimNode*> gpu_node, host_node;

		/* visualization */
		SGUCHAR *dev_visual, *host_visual;

		/* temporary buffers for some purpose */
		double *dev_dtpbuf, *host_dtpbuf;
		int    *dev_ntpbuf, *host_ntpbuf;

		/* cursor */
		int3 m_cursor;
		
		/* CUDA */
		dim3 gridDim, blockDim;

		/* node and volumetric size */
		size_t m_node_size, m_volm_size;

		/* title bar */
		string m_sz_title;

		/* etc. */
		int increase_times, decrease_times;
		SimNode *ptr;

	private:
		FunctionHelper helper;

#pragma endregion

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

	public:
		void ZeroBuffers( void );
		sstr GetTitleBar( void );
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
		bool AllocateResource( void );
		bool allocHostRes( void );
		bool allocDeviceRes( void );
		bool allocShareBuffers( void );
		bool allocVisualBuffers( void );
		void allocTopologyNodes( void );

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