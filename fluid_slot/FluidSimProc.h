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
#define dev_slot_d           dev_slot[ 0 ]
#define dev_slot_d0          dev_slot[ 1 ]
#define dev_slot_u           dev_slot[ 2 ]
#define dev_slot_u0          dev_slot[ 3 ]
#define dev_slot_v           dev_slot[ 4 ]
#define dev_slot_v0          dev_slot[ 5 ]
#define dev_slot_w           dev_slot[ 6 ]
#define dev_slot_w0          dev_slot[ 7 ]
#define dev_slot_div         dev_slot[ 8 ]
#define dev_slot_p           dev_slot[ 9 ]
#define dev_slot_obs         dev_slot[ 10 ]


#define dev_bufs_num         28
#define dev_dens_C           dev_bufs[ 0 ]
#define dev_dens_L           dev_bufs[ 1 ]
#define dev_dens_R           dev_bufs[ 2 ]
#define dev_dens_U           dev_bufs[ 3 ]
#define dev_dens_D           dev_bufs[ 4 ]
#define dev_dens_F           dev_bufs[ 5 ]
#define dev_dens_B           dev_bufs[ 6 ]

#define dev_velu_C           dev_bufs[ 7 ]
#define dev_velu_L           dev_bufs[ 8 ] 
#define dev_velu_R           dev_bufs[ 9 ]
#define dev_velu_U           dev_bufs[ 10 ]
#define dev_velu_D           dev_bufs[ 11 ]
#define dev_velu_F           dev_bufs[ 12 ]
#define dev_velu_B           dev_bufs[ 13 ]

#define dev_velv_C           dev_bufs[ 14 ]
#define dev_velv_L           dev_bufs[ 15 ]
#define dev_velv_R           dev_bufs[ 16 ]
#define dev_velv_U           dev_bufs[ 17 ]
#define dev_velv_D           dev_bufs[ 18 ]
#define dev_velv_F           dev_bufs[ 19 ]
#define dev_velv_B           dev_bufs[ 20 ]

#define dev_velw_C           dev_bufs[ 21 ]
#define dev_velw_L           dev_bufs[ 22 ]
#define dev_velw_R           dev_bufs[ 23 ]
#define dev_velw_U           dev_bufs[ 24 ]
#define dev_velw_D           dev_bufs[ 25 ]
#define dev_velw_F           dev_bufs[ 26 ]
#define dev_velw_B           dev_bufs[ 27 ]

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
		/* GPU slot */
		vector <double*> dev_slot;
		/* level-S GPU nodes */
		vector <double*> dev_densS;
		vector <double*> dev_veluS;
		vector <double*> dev_velvS;
		vector <double*> dev_velwS;
		/* level-T GPU nodes */
		vector <double*> dev_densT;
		vector <double*> dev_veluT;
		vector <double*> dev_velvT;
		vector <double*> dev_velwT;
		/* GPU obstacle */
		vector <double*> dev_obst;
		/* temporary buffers */
		vector <double*> dev_bufs;

		/* Host nodes */
		vector <double*> host_dens;   // density
		vector <double*> host_velu;   // velocity u
		vector <double*> host_velv;   // velocity v
		vector <double*> host_velw;   // velocity w
		vector <double*> host_obst;    // obstacle
		vector <SimNode*> host_node;

		/* visualization */
		SGUCHAR *dev_visual, *host_visual;

	private:
		FunctionHelper helper;

	private:
		/* size for allocation memories */
		size_t m_node_size;
		size_t m_volm_size;
		size_t m_slot_size;

		/* for add source */
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
		/* initialize FPS and etc. */
		void InitParams( FLUIDSPARAM *fluid );

		/* copy host data to CUDA device */
		void LoadBullet( int i, int j, int k );

		/* retrieve data back to host */
		void ExitBullet( int i, int j, int k );

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