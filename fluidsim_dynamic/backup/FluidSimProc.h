/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 14, 2014
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
#include "DataStructures.h"

using std::vector;

namespace sge
{
	/* fluid simulation */
	class FluidSimProc
	{
	private:
		struct LinkNode
		{
			int3 n3Pos;
			bool active;
			LinkNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack; 
		};

	/****************************************************************/
	private: // fluid simulation buffers		
		/* GPU buffers */
		vector<double*> dev_buffers;

		/* host buffers */
		vector<double*> host_density;
		vector<double*> host_velocity_u;
		vector<double*> host_velocity_v;
		vector<double*> host_velocity_w;
		vector<double*> host_obstacle;

		/* linking message */
		vector<LinkNode*> host_link;

		/* Level-0 host volume rendering buffers */
		SGUCHAR *host_visual, *dev_visual;
	/****************************************************************/
	
	/****************************************************************/
	private: // etc.
		/* target */
		SGINT3 nodeIX;

		/* etc */
		FunctionHelper m_helper;
	/****************************************************************/

	public:
		/* default constructor */
		FluidSimProc( FLUIDSPARAM *fluid );

		/* fluid simulation processing function */
		void FluidSimSolver( FLUIDSPARAM *fluid );

		/* allocate resource */
		bool AllocateResource( void );

		/* when program existed, release resource */
		void FreeResource( void );

		/* zero the buffers for fluid simulation */
		void ZeroBuffers( void );

		/* mark the node as actived */
		void ActiveNode( int i, int j, int k );

		/* mark the node as deactived */
		void DeactiveNode( int i, int j, int k );

		/* select a node */
		void SelectNode( int i, int j, int k );

		/* retrieve the density back and load into volumetric data for rendering */
		void PickVolumetric( FLUIDSPARAM *fluid );

	private:
		/* copy host data to CUDA device */
		void UploadBuffers( void );

		/* retrieve data back to host */
		void DownloadBuffers( void );

		/* initialize the fluid simulation parameters, such as FPS and etc. */
		void InitParams( FLUIDSPARAM *fluid );

		/* zero data, set the bounds */
		void InitSimNodes( void );

		/* create simulation nodes' topological structure */
		void BuildOrder( void );

		/* add source */
		void AddSource( void );
	};
};

#define dev_buffers_num         12
#define dev_dens    dev_buffers[0]
#define dev_dens0   dev_buffers[1]
#define dev_vel_u   dev_buffers[2]
#define dev_vel_v   dev_buffers[3]
#define dev_vel_w   dev_buffers[4]
#define dev_vel_u0  dev_buffers[5]
#define dev_vel_v0  dev_buffers[6]
#define dev_vel_w0  dev_buffers[7]
#define dev_div     dev_buffers[8]
#define dev_p       dev_buffers[9]
#define dev_obs     dev_buffers[10]

#define dev_center  dev_buffers[0]
#define dev_left    dev_buffers[1]
#define dev_right   dev_buffers[2]
#define dev_up      dev_buffers[3]
#define dev_down    dev_buffers[4]
#define dev_front   dev_buffers[5]
#define dev_back    dev_buffers[6]

#define BOUND_BLANK   0.f
#define BOUND_SOURCE  1.f
#define BOUND_WALL    100.f

#endif