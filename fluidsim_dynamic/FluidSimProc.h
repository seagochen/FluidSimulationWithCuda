/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 20, 2014
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

#define dev_buffers_num                   18
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

#define dev_center           dev_buffers[ 11 ]
#define dev_left             dev_buffers[ 12 ]
#define dev_right            dev_buffers[ 13 ]
#define dev_up               dev_buffers[ 14 ]
#define dev_down             dev_buffers[ 15 ]
#define dev_front            dev_buffers[ 16 ]
#define dev_back             dev_buffers[ 17 ]

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
		size_t m_node_size;
		size_t m_volm_size;
		double *dev_tpbufs, *host_tpbufs;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

		/* fluid simulation processing function */
		void FluidSimSolver( FLUIDSPARAM *fluid );

		/* when program existed, release resource */
		void FreeResource( void );

		/* zero the buffers for fluid simulation */
		void ZeroBuffers( void );

	private:
		/* upload neighbouring buffers to GPU device */
		void UploadNeighbouringBuffers( vector<double*> container, int i, int j, int k );

		/* download neightbouring buffers to host */
		void DownloadNeighbouringBuffers( vector<double*> container, int i, int j, int k );

		/* flood buffer for multiple nodes */
		void TracingDensity( void );

		/* data flooding */
		void DataFlooding( vector<double*> container, int i, int j, int k, bool isDensity );

		/* initialize FPS and etc. */
		void InitParams( FLUIDSPARAM *fluid );

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
		void InitBoundary( int i, int j, int k );

		/* solving velocity */
		void VelocitySolver ( void );
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

#endif