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
	/****************************************************************/
	private: // fluid simulation buffers		
		/* Level-0 GPU buffers */
		vector<SGSIMPLENODES*> dev_L0_vector;

		/* Level-1 GPU buffers */
		SGCUDANODES *dev_L1_bufs;

		/* Level-2 GPU buffers */
		vector<SGSTDGRID*> dev_L2_vector;

		/* Level-0 host buffers */
		vector<SGHOSTNODE> host_L0_vector;

		/* Level-1 host volume rendering buffers */
		SGUCHAR *host_L0_visual, *dev_L0_visual;
	/****************************************************************/

	/****************************************************************/
	private: // volumetric rendering buffers
		/* volumetric rendering data */
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