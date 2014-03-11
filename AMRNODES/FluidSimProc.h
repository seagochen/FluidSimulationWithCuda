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

		/* when program existed, release resource */
		void FreeResource( void );

		/* zero the buffers for fluid simulation */
		void ZeroBuffers( void );

		/* title bar */
		ptrStr GetTitleBar( void );

		/* print runtime message */
		void PrintMSG( void );

		/* upload buffers */
		void IO_UploadBuffers( void );

		/* download buffers */
		void IO_DownloadBuffers( void );

	private:
		void IO_ReadBuffers( void ); // IO, host to device
		void IO_WriteBuffers( void ); // IO, device to host
		void LoadNode( int i, int j, int k );  // loading gpu nodes for fluid simulation
		void SaveNode( int i, int j, int k ); // saving the result of fluid simulation
		void SolveNavierStokers( void ); // solving the Navier-Stokers equations
		void Interaction( int i, int j, int k ); // flood buffer for multiple nodes
		void InitParams( FLUIDSPARAM *fluid ); // initialize FPS and etc.
		void InitBoundary( void ); 	// initialize boundary condition

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
		void DensitySolver( void );
		void AddSource( void );
		void VelocitySolver( void );
		void Projection( void );
		void Diffusion( double *grid_out, cdouble *grid_in, cdouble diffusion, cdouble *obstacle, cint field );
		void Advection( double *grid_out, cdouble *grid_in, cdouble *obstacle, cint field, cdouble *u_in, cdouble *v_in, cdouble *w_in );
		void Jacobi( double *grid_out, cdouble *grid_in, cdouble *obstacle, cint field, cdouble diffusion, cdouble divisor );
	};
};

#endif