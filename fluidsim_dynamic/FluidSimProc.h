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

#define dev_buffers_num                 11
#define dev_u                dev_buffers [ 0 ]
#define dev_v                dev_buffers [ 1 ]
#define dev_w                dev_buffers [ 2 ]
#define dev_u0               dev_buffers [ 3 ]
#define dev_v0               dev_buffers [ 4 ]
#define dev_w0               dev_buffers [ 5 ]
#define dev_den              dev_buffers [ 6 ]
#define dev_den0             dev_buffers [ 7 ]
#define dev_div              dev_buffers [ 8 ]
#define dev_p                dev_buffers [ 9 ]
#define dev_obs              dev_buffers [ 10 ]

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

		void FluidSimSolver( FLUIDSPARAM *fluid );
		void FreeResource( void );
		void ZeroBuffers( void );

	private:
		void InitFPS( FLUIDSPARAM *fluid );
		void NodetoDevice( void );
		void DevicetoNode( void );
		void SelectNode( int i, int j, int k );
		bool ActiveNode( int i, int j, int k );
		bool DeactiveNode( int i, int j, int k );
		void GetDensityImage( void );
		void GenerateVolumeData( FLUIDSPARAM *fluid );

	private:
		SGRUNTIMEMSG AllocateResource( FLUIDSPARAM *fluid );
		void DensitySolver ( void );
		void VelocitySolver ( void );
	};
};

#endif