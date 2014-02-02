/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 02, 2014
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
		/* vector list */
		vector<SGHOSTNODE> host_nodes;
		vector<SGDOUBLE*>  dev_buf;

		/* buffer for volume rendering */
		SGUCHAR *host_visual, *dev_visual;

		/* buffer for temporary storing */
		double *dev_fbuf;

		/* fluid simulation buffers */
		SGDEVICEBUFF *dev_global;

		/* target */
		int nodeIX;

		/* etc */
		FunctionHelper m_helper;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

		void FluidSimSolver( FLUIDSPARAM *fluid );
		void FreeResource( void );
		void ZeroAllBuffer( void );
		void ZeroDevData( void );
		void SelectNode( int i, int j, int k );
		void SelectNode( int index );

	private:
		SGRUNTIMEMSG AllocateResourcePtrs( FLUIDSPARAM *fluid );
		void PickData( FLUIDSPARAM *fluid );
		void CopyDataToHost( void );
		void CopyDataToDevice( void );
		void SetObstacle( void );
		void LinkDataset( void );
		void SetParameters( FLUIDSPARAM *fluid );

	private:
		/* building nodes-structure */
		void BuildStructure( void );
		/* left */
		void LeftDataToHost( void );
		void LeftDataToDevice( void );
		/* right */
		void RightDataToHost( void );
		void RightDataToDevice( void );
		/* up */
		void UpDataToHost( void );
		void UpDataToDevice( void );
		/* down */
		void DownDataToHost( void );
		void DownDataToDevice( void );
		/* front */
		void FrontDataToHost( void );
		void FrontDataToDevice( void );
		/* back */
		void BackDataToHost( void );
		void BackDataToDevice( void );
	};
};

#endif