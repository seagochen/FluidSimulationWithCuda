/**
* <Author>      Orlando Chen
* <First>       Dec 15, 2013
* <Last>		Jan 22, 2014
* <File>        FluidSimAreaDynamic.h
*/

#ifndef __fluid_sim_area_dynamic_h_
#define __fluid_sim_area_dynamic_h_

#include <vector>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <Windows.h>

#define DELTATIME           0.3f
#define STEPSIZE            0.001f
#define DIFFUSION           0.1f
#define VISOCITY            0.0f 
#define SOURCE              30

#define GRIDS_X             64
#define NODES_X             3
#define VOLUME_X            GRIDS_X*NODES_X
#define THREADS_X           512
#define TILE_X              16
#define WINDOWS_X           600
#define CANVAS_X            600
#define CUBESIZE_X          GRIDS_X*GRIDS_X*GRIDS_X
#define FACESIZE_X          GRIDS_X*GRIDS_X
#define TPBUFFER_X          1024

#define DevGridsNum         32
#define dev_u               dev_grids[ 0 ] // velocity u
#define dev_v               dev_grids[ 1 ] // velocity v
#define dev_w               dev_grids[ 2 ] // velocity w
#define dev_d               dev_grids[ 3 ] // density
#define dev_o               dev_grids[ 4 ] // obstacle
#define dev_div             dev_grids[ 5 ] // divergence
#define dev_p               dev_grids[ 6 ] // projection
#define dev_t               dev_grids[ 7 ] // temporary

#define dev_d_U             dev_grids[ 8 ] // up cell
#define dev_u_U             dev_grids[ 9 ] //
#define dev_v_U             dev_grids[ 10 ]//
#define dev_w_U             dev_grids[ 11 ]//

#define dev_d_D             dev_grids[ 12 ] // down cell
#define dev_u_D             dev_grids[ 13 ] //
#define dev_v_D             dev_grids[ 14 ] //
#define dev_w_D             dev_grids[ 15 ] //

#define dev_d_L             dev_grids[ 16 ] // left cell
#define dev_u_L             dev_grids[ 17 ] //
#define dev_v_L             dev_grids[ 18 ] //
#define dev_w_L             dev_grids[ 19 ] //

#define dev_d_R             dev_grids[ 20 ] // right cell
#define dev_u_R             dev_grids[ 21 ] //
#define dev_v_R             dev_grids[ 22 ] //
#define dev_w_R             dev_grids[ 23 ] //

#define dev_d_F             dev_grids[ 24 ] // front cell
#define dev_u_F             dev_grids[ 25 ] //
#define dev_v_F             dev_grids[ 26 ] //
#define dev_w_F             dev_grids[ 27 ] //

#define dev_d_B             dev_grids[ 28 ] // back cell
#define dev_u_B             dev_grids[ 29 ] //
#define dev_v_B             dev_grids[ 30 ] //
#define dev_w_B             dev_grids[ 31 ] //

#define BD_SOURCE          -1
#define BD_BLANK            0
#define BD_HLAF             5
#define BD_WALL             10
#define BD_EXIT             100

/*
  -------------------------------------------------------------------------------------------------------
   Dataset
  -------------------------------------------------------------------------------------------------------
*/

typedef GLuint handler;
using std::vector;

namespace sge
{
	/* structure of fluidsim */
	struct fluidsim
	{
#pragma region definition of structures, such as shader, textures, volume, fps and etc.
		static struct shader
		{
			handler hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
			char   *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
			Shader *ptrShader;
		};
		
		struct textures
		{
			handler  hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
		};
		
		struct volume
		{
			uchar  *ptrData;
			size_t  uWidth, uHeight, uDepth;
		};
		
		typedef struct raycasting
		{
			handler hCluster;
			int     nAngle;
			boolean bRun;
			GLfloat fStepsize;
			size_t  uCanvasWidth, uCanvasHeight;
		}rayc;
		
		struct thread
		{
			DWORD   dwThreadId;
			HANDLE  hThread;
		};
		
		struct fps
		{
			DWORD dwFrames;
			DWORD dwCurrentTime;
			DWORD dwLastUpdateTime;
			DWORD dwElapsedTime;
			UINT  uFPS;
		};
#pragma endregion
		
		shader   shader;
		textures textures;
		volume   volume;
		rayc     ray;
		thread   thread;
		fps      fps;
	};

	/* fluid simulation processor */
	class FluidSimProc
	{
		struct node
		{
			node    *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
			boolean  bActive;
			int      i, j, k;
			double  *ptrDens, *ptrVelU, *ptrVelV, *ptrVelW, *ptrObs;
		};

	private:
		/* vector list */
		vector<double*> dev_grids;
		vector<node>    host_nodes;

		/* buffer for volume rendering */
		uchar *host_visual, *dev_visual;

		/* buffer for temporary storing */
		int     *host_ibuf, *dev_ibuf;
		double  *host_fbuf, *dev_fbuf;

		/* target */
		int m_index;

	public:
		FluidSimProc( fluidsim *fluid );

		void FluidSimSolver( fluidsim *fluid );
		void FreeResourcePtrs( void );
		void ZeroAllBuffer( void );
		void ZeroDevData( void );
		void SelectNode( int i, int j, int k );
		void SelectNode( int index );

	private:
		SGRUNTIMEMSG AllocateResourcePtrs( fluidsim *fluid );
		void DensitySolver( void );
		void VelocitySolver( void );
		void PickData( fluidsim *fluid );
		void CopyDataToHost( void );
		void CopyDataToDevice( void );
		void SetObstacle( void );
		void LinkDataset( void );
		void SetParameters( fluidsim *fluid );

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


#pragma region cuda controllers

/*
  -------------------------------------------------------------------------------------------------------
   Custom CUDA Functions
  -------------------------------------------------------------------------------------------------------
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define cudaDevice(gridDim,blockDim) <<<gridDim,blockDim>>>

#define cudaIndex2D(i,j,elements_x) ((j)*(elements_x)+(i))

#define cudaIndex3D(i,j,k,elements_x) ((k)*elements_x*elements_x+(j)*elements_x+(i))

#define Index(i,j,k) cudaIndex3D(i,j,k,GRIDS_X)

#define cudaTrans2DTo3D(i,j,k,elements_x) \
	k = cudaIndex2D(i,j,(elements_x)) / ((elements_x)*(elements_x)); \
	i = i % (elements_x); \
	j = j % (elements_x); \

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = TILE_X; \
	blockDim.y = TILE_X; \
	gridDim.x  = GRIDS_X / TILE_X; \
	gridDim.y  = GRIDS_X / TILE_X; \

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (GRIDS_X / TILE_X); \
	blockDim.y = (THREADS_X / TILE_X); \
	gridDim.x  = (GRIDS_X / blockDim.x); \
	gridDim.y  = (GRIDS_X * GRIDS_X * GRIDS_X) / (blockDim.x * blockDim.y * (GRIDS_X / blockDim.x)); \

#define GetIndex()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D ( i, j, k, GRIDS_X ); \

/*
  -------------------------------------------------------------------------------------------------------
   Define something
  -------------------------------------------------------------------------------------------------------
*/

#include "iso646.h"

#define gst_header        0              /* (ghost, halo) the header cell of grid */
#define sim_header        1              /* (actually) the second cell of grid */
#define gst_tailer       GRIDS_X - 1    /* (ghost, halo) the last cell of grid */
#define sim_tailer       GRIDS_X - 2    /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_tailer ) \
	if ( j >= sim_header and j <= sim_tailer ) \
	if ( k >= sim_header and k <= sim_tailer ) {

#define EndSimArea() }

#pragma endregion

#endif