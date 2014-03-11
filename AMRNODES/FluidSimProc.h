/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 11, 2014
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
	private:

		struct SimNode
		{
			SGBOOLEAN updated;
			int3 nodeIX;
			SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		};

	private:

		/* temporary buffers for fluid simulation */
		vector <double*> dev_buffers;

		/* nodes for gpu and host */
		vector <double*> dev_density,    host_density;
		vector <double*> dev_velocity_u, host_velocity_u;
		vector <double*> dev_velocity_v, host_velocity_v;
		vector <double*> dev_velocity_w, host_velocity_w;
		vector <double*> dev_obstacle,   host_obstacle;

		/* local nodes, the last node is the global */
		vector <double*> node_density;
		vector <double*> node_velocity_u;
		vector <double*> node_velocity_v;
		vector <double*> node_velocity_w;
		vector <double*> node_obstacle;

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

	private:
		FunctionHelper helper;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

		/* fluid simulation processing function */
		void FluidSimSolver( FLUIDSPARAM *fluid );

	private:
		void ReadBuffers( void );  // load host copy to device
		void WriteBuffers( void ); // save device to host copy
		void ClearBullet( void );
		void LoadBullet( int i, int j, int k );
		void ExitBullet( int i, int j, int k );

	private:
		void SolveRootGrids( void );
		void SolveLeafGrids( void );
		void InterRootGrids( void );
		void InterLeafGrids( void );
		void Interaction( int i, int j, int k );
		void SolveNavierStokers( void );
		void SetCursor( int i, int j, int k );				

	public:
		void FreeResource( void );
		void ZeroBuffers( void );
		sstr GetTitleBar( void );
		void PrintMSG( void );
		void HostToDevice( void );
		void DeviceToHost( void );
		void InitParams( FLUIDSPARAM *fluid );
		void InitBoundary( void );

	private:
		void CreateTopology( void );
		void CreateHostTopology( void );
		void CreateDeviceTopology( void );

	private:
		void ZeroHostBuffers( void );
		void ZeroDeviceBuffers( void );
		void ZeroTempBuffers( void );
		void ZeroVolumeBuffers( void );

	private:
		void RefreshStatus( FLUIDSPARAM *fluid );
		void RefreshFPS( FLUIDSPARAM *fluid );
		void RefreshVolume( FLUIDSPARAM *fluid );
		void RefreshHostNodes( FLUIDSPARAM *fluid );

	private:
		bool AllocateResource( void );
		bool CreateHostNodes( void );
		bool CreateDeviceNodes( void );
		bool CreateTempBuffers( void );
		bool CreateVolumeBuffers( void );

	private:
		void FreeHostNodes( void );
		void FreeDeviceNodes( void );
		void FreeTempBuffers( void );
		void FreeVolumeBuffers( void );

	private:
		void AddSource( void );
		void DensitySolver( cdouble delta );
		void VelocitySolver( cdouble delta );
		void Projection( void );
		void Diffusion( double *out, cdouble *in, cdouble diffusion, cdouble *obstacle, cint field );
		void Advection( double *out, cdouble *in, cdouble delta, cdouble *obstacle, cint field, cdouble *u, cdouble *v, cdouble *w );
		void Jacobi( double *out, cdouble *in, cdouble *obstacle, cint field, cdouble diffusion, cdouble divisor );
	};
};

#endif