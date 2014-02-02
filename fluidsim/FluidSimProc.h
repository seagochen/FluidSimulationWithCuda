/**
* <Author>      Orlando Chen
* <First>       Dec 15, 2013
* <Last>		Feb 02, 2014
* <File>        FluidSimProc.h
*/

#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "CUDADef.h"
#include "CUDAFunctionHelper.h"
#include "DataStructures.h"

#define DEVLISTNUM       10
#define dev_u     dev_buf[0]
#define dev_v     dev_buf[1]
#define dev_w     dev_buf[2]
#define dev_div   dev_buf[3]
#define dev_p     dev_buf[4]
#define dev_dens  dev_buf[5]
#define dev_u0    dev_buf[6]
#define dev_v0    dev_buf[7]
#define dev_w0    dev_buf[8]
#define dev_dens0 dev_buf[9]

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
		CUDAFuncHelper cudahelper;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

		void FluidSimSolver( FLUIDSPARAM *fluid );
		void FreeResourcePtrs( void );
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