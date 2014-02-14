/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 14, 2014
* <File Name>     FluidSimProc.h
*/


#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

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
		/* nodes, and buffers for fluid simulation */
		vector<SGHOSTNODE> host_nodes;
		vector<SGTEMPBUFFERS*> dev_bufs;

		/* fluid simulation buffers */
		SGCUDANODES *dev_nodes;
		SGSTDGRID *ptrCenter, *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		
		/* buffer for temporary storing */
		SGDOUBLE  *dev_stores;

		/* buffer for volume rendering */
		SGUCHAR *host_visual, *dev_visual;
	
	private:
		/* target */
		int m_ix;

		/* etc */
		FunctionHelper m_helper;

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

		/* choose the node and mark it as actived */
		void ActiveNode( int i, int j, int k );

		/* choose the node and mark it as deactived */
		void DeactiveNode( int i, int j, int k );

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
	};
};

#endif