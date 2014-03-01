/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 01, 2014
* <File Name>     FluidSimProc.h
*/


#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "CUDAMacroDef.h"
#include "FunctionHelper.h"

#define dev_slot_num         11
#define dev_den              dev_slot[ 0 ]
#define dev_den0             dev_slot[ 1 ]
#define dev_u                dev_slot[ 2 ]
#define dev_u0               dev_slot[ 3 ]
#define dev_v                dev_slot[ 4 ]
#define dev_v0               dev_slot[ 5 ]
#define dev_w                dev_slot[ 6 ]
#define dev_w0               dev_slot[ 7 ]
#define dev_div              dev_slot[ 8 ]
#define dev_p                dev_slot[ 9 ]
#define dev_obs              dev_slot[ 10 ]

#define dev_buffers_num      35
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

using std::vector;

namespace sge
{
	struct SimNode
	{
		SGINT3 nodeIX;
		SGBOOLEAN active;
		SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};

	class FluidSimProc
	{
	private:
		vector <double*> dev_slot;
		vector <double*> dev_buffers;
		vector <double*> dev_density;
		vector <double*> dev_velocity_u;
		vector <double*> dev_velocity_v;
		vector <double*> dev_velocity_w;
		vector <double*> dev_obstacle;

		vector <double*> host_density;
		vector <double*> host_velocity_u;
		vector <double*> host_velocity_v;
		vector <double*> host_velocity_w;
		vector <double*> host_obstacle;
		vector <SimNode*> host_node;

		SGUCHAR *dev_visual, *host_visual;

	private:
		FunctionHelper helper;

	private:
		size_t m_node_size;
		size_t m_volm_size;
		size_t m_slot_size;
		double *dev_tpbufs, *host_tpbufs;
		int    increase_times, decrease_times;
		double dTimes;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

		/* fluid simulation processing function */
		void FluidSimSolver( FLUIDSPARAM *fluid );

		/* when program existed, release resource */
		void FreeResource( void );

		/* zero the buffers for fluid simulation */
		void ZeroBuffers( void );

	private:
		/* flood buffer for multiple nodes */
		void TracingTheFlow( int i, int j, int k  );

		/* initialize FPS and etc. */
		void InitParams( FLUIDSPARAM *fluid );

		/* copy host data to CUDA device */
		void NodeToDevice( int i, int j, int k );

		/* retrieve data back to host */
		void DeviceToNode( int i, int j, int k );

		/* mark the node as actived */
		bool ActiveTheNode( int i, int j, int k );

		/* mark the node as deactived */
		bool DeactiveTheNode( int i, int j, int k );
				
		/* retrieve the density back and load into volumetric data for rendering */
		void GetVolumetric( FLUIDSPARAM *fluid );
		
		/* create simulation nodes' topological structure */
		void BuildOrder( void );

		/* allocate resource */
		bool AllocateResource( FLUIDSPARAM *fluid );

		/* solving density */
		void DensitySolver( void );

		/* add source */
		void AddSource( void );

		/* initialize boundary condition */
		void InitBoundary( int i, int j, int k );

		/* solving velocity */
		void VelocitySolver( void );

		/* upload all buffers */
		void UploadBuffers( void );
	};
};

#define MACRO_DENSITY     0
#define MACRO_VELOCITY_U  1
#define MACRO_VELOCITY_V  2
#define MACRO_VELOCITY_W  3
#define MACRO_SIMPLE      4

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_SOURCE     1
#define MACRO_BOUNDARY_OBSTACLE 100

#define MACRO_CENTER     0
#define MACRO_LEFT       1
#define MACRO_RIGHT      2
#define MACRO_UP         3
#define MACRO_DOWN       4
#define MACRO_FRONT      5
#define MACRO_BACK       6

#define TESTING_MODE_SWITCH     0 /* switch: close(0) open(1) */
#define TESTING_MODE            0 /* velocity: default-up(0) down(1) left(2) right(3) front(4) back(5) */

#endif