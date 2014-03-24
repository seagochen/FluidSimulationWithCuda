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

#define host_density    m_vectHostDens
#define host_velocity_u m_vectHostVelU
#define host_velocity_v m_vectHostVelV
#define host_velocity_w m_vectHostVelW
#define host_obstacle   m_vectHostObst

#define dev_density     m_vectGPUDens
#define dev_velocity_u  m_vectGPUVelU
#define dev_velocity_v  m_vectGPUVelV
#define dev_velocity_w  m_vectGPUVelW
#define dev_obstacle    m_vectGPUObst

#define node_density    m_vectNewDens
#define node_velocity_u m_vectNewVelU
#define node_velocity_v m_vectNewVelV
#define node_velocity_w m_vectNewVelW
#define node_obstacle   m_vectNewObst

#define dev_buffers     m_vectCompBufs
	
	class FluidSimProc
	{
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
		string m_szTitle;

		/* etc. */
		int increase_times, decrease_times;
		SimNode *ptr;

	private:
		FunctionHelper helper;
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

		void CreateTopology( void );

		void RefreshStatus( FLUIDSPARAM *fluid );

		void FluidSimSolver( FLUIDSPARAM *fluid );

	private:
		void SolveNavierStokesEquation( cdouble timestep, bool add );
		void DensitySolver( cdouble timestep );
		void VelocitySolver( cdouble timestep );
		void SourceSolver( void );
		void InitBoundary( void );
		void Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor );
		void Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w );
		void Diffusion( double *out, cdouble *in, cdouble diff );
		void Projection( double *u, double *v, double *w, double *div, double *p );
	};
};

#define __device_func__ <<<gridDim,blockDim>>>
#define DeviceParamDim() m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z )
#define _zero(vect) kernelZeroBuffers __device_func__ ( vect, GRIDS_X, GRIDS_Y, GRIDS_Z )

#define dev_buffers_num                   35
#define dev_den              dev_buffers[ 0 ]
#define dev_den0             dev_buffers[ 1 ]
#define dev_u                dev_buffers[ 2 ]
#define dev_u0               dev_buffers[ 3 ]
#define dev_v                dev_buffers[ 4 ]
#define dev_v0               dev_buffers[ 5 ]
#define dev_w                dev_buffers[ 6 ]
#define dev_w0               dev_buffers[ 7 ]
#define dev_div              dev_buffers[ 8 ]
#define dev_p                dev_buffers[ 9 ]
#define dev_obs              dev_buffers[ 10 ]

#define dens_C               dev_buffers[ 0 ]
#define dens_L               dev_buffers[ 11 ]
#define dens_R               dev_buffers[ 12 ]
#define dens_U               dev_buffers[ 13 ]
#define dens_D               dev_buffers[ 14 ]
#define dens_F               dev_buffers[ 15 ]
#define dens_B               dev_buffers[ 16 ]

#define velu_C               dev_buffers[ 2 ]
#define velu_L               dev_buffers[ 17 ] 
#define velu_R               dev_buffers[ 18 ]
#define velu_U               dev_buffers[ 19 ]
#define velu_D               dev_buffers[ 20 ]
#define velu_F               dev_buffers[ 21 ]
#define velu_B               dev_buffers[ 22 ]

#define velv_C               dev_buffers[ 4 ]
#define velv_L               dev_buffers[ 23 ]
#define velv_R               dev_buffers[ 24 ]
#define velv_U               dev_buffers[ 25 ]
#define velv_D               dev_buffers[ 26 ]
#define velv_F               dev_buffers[ 27 ]
#define velv_B               dev_buffers[ 28 ]

#define velw_C               dev_buffers[ 6 ]
#define velw_L               dev_buffers[ 29 ]
#define velw_R               dev_buffers[ 30 ]
#define velw_U               dev_buffers[ 31 ]
#define velw_D               dev_buffers[ 32 ]
#define velw_F               dev_buffers[ 33 ]
#define velw_B               dev_buffers[ 34 ]

#endif