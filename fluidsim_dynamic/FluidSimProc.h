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

#define dev_buffers_num                   11
#define dev_den              dev_buffers[ 0 ]
#define dev_center           dev_buffers[ 0 ]
#define dev_den0             dev_buffers[ 1 ]
#define dev_u                dev_buffers[ 2 ]
#define dev_left             dev_buffers[ 2 ]
#define dev_u0               dev_buffers[ 3 ]
#define dev_v                dev_buffers[ 4 ]
#define dev_right            dev_buffers[ 4 ]
#define dev_v0               dev_buffers[ 5 ]
#define dev_w                dev_buffers[ 6 ]
#define dev_up               dev_buffers[ 6 ]
#define dev_w0               dev_buffers[ 7 ]
#define dev_div              dev_buffers[ 8 ]
#define dev_down             dev_buffers[ 8 ]
#define dev_p                dev_buffers[ 9 ]
#define dev_front            dev_buffers[ 9 ]
#define dev_obs              dev_buffers[ 10 ]
#define dev_back             dev_buffers[ 10 ]


using std::vector;

namespace sge
{
	class FluidSimProc
	{
	private:
		struct SimNode
		{
			SGINT3 nodeIX;
			SGBOOLEAN active;
			SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		};

	private:
		vector <double*> dev_buffers;

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
		SGINT3 nPos;

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
		void FloodBuffers( void );

		/* flood density */
		void FloodDensityBuffers( void );

		/* flood velocity u */
		void FloodVelocityBuffersU( void );

		/* flood velocity v */
		void FloodVelocityBuffersV( void );

		/* flood velocity w */
		void FloodVelocityBuffersW( void );

		/* initialize FPS and etc. */
		void InitFPS( FLUIDSPARAM *fluid );

		/* copy host data to CUDA device */
		void NodetoDevice( void );

		/* retrieve data back to host */
		void DevicetoNode( void );

		/* select a node */
		bool SelectNode( int i, int j, int k );

		/* mark the node as actived */
		bool ActiveNode( int i, int j, int k );

		/* mark the node as deactived */
		bool DeactiveNode( int i, int j, int k );
		
		/* cast density to volumetric data */
		void DensitytoVolumetric( void );
		
		/* retrieve the density back and load into volumetric data for rendering */
		void GetVolumetric( FLUIDSPARAM *fluid );
		
		/* create simulation nodes' topological structure */
		void BuildOrder( void );

		/* allocate resource */
		SGRUNTIMEMSG AllocateResource( FLUIDSPARAM *fluid );

		/* solving density */
		void DensitySolver( void );

		/* add source */
		void AddSource( void );

		/* initialize boundary condition */
		void InitBoundary( void );

		/* solving velocity */
		void VelocitySolver ( void );
	};
};

#endif