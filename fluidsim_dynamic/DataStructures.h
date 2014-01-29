/**
* <Author>      Orlando Chen
* <First>       Jan 23, 2014
* <Last>		Jan 28, 2014
* <File>        DataStructures.h
*/

#ifndef __data_and_structures_h_
#define __data_and_structures_h_

#include <cuda_runtime.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#define DELTATIME           0.5f
#define STEPSIZE            0.001f
#define DIFFUSION           0.1f
#define VISOCITY            0.0f 
#define SOURCE_DENSITY      30
#define SOURCE_VELOCITY     15

#define GRIDS_X             64
#define NODES_X             3
#define VOLUME_X            GRIDS_X*NODES_X
#define THREADS_X           512
#define TILE_X              16
#define WINDOWS_X           600
#define CANVAS_X            600
#define CUBESIZE_X          GRIDS_X*GRIDS_X*GRIDS_X
#define TPBUFFER_X          1024

namespace sge
{
	typedef double3 SGDOUBLE3;
	typedef double4 SGDOUBLE4;
	typedef int3    SGINT3;
	typedef int4    SGINT4;

	enum SGBOUNDARY
	{
		SG_BD_SOURCE,
		SG_BD_BLANK,
		SG_BD_WALL,
	};
	
	enum SGGRIDTYPE
	{
		SG_DENS_GRID,
		SG_VELU_GRID,
		SG_VELV_GRID,
		SG_VELW_GRID,
		SG_DIV_GRID,
		SG_PRES_GRID,
	};
	
	enum SGNODECODE
	{
		SG_USING_CENTER,
		SG_USING_LEFT,
		SG_USING_RIGHT,
		SG_USING_UP,
		SG_USING_DOWN,
		SG_USING_FRONT,
		SG_USING_BACK,
		SG_NO_DEFINE,
	};
	
	enum SGJACOBITYPE
	{
		SG_SOLVE_DENSITY,
		SG_SOLVE_VELOCITY,
	};

	typedef struct FLUIDSPARAM
	{
		struct SHADER
		{
			SGHANDLER hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
			SGCHAR   *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
			SGSHADER *ptrShader;
		};
		
		struct TEXTURES
		{
			SGHANDLER  hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
		};
		
		struct VOLUME
		{
			SGUCHAR *ptrData;
			size_t   uWidth, uHeight, uDepth;
		};
		
		struct RAYCAST
		{
			SGHANDLER hCluster;
			SGINT     nAngle;
			SGBOOLEAN bRun;
			SGFLOAT   fStepsize;
			size_t    uCanvasWidth, uCanvasHeight;
		};
		
		struct THREAD
		{
			DWORD   dwThreadId;
			HANDLE  hThread;
		};
		
		struct FPS
		{
			DWORD dwFrames;
			DWORD dwCurrentTime;
			DWORD dwLastUpdateTime;
			DWORD dwElapsedTime;
			UINT  uFPS;
		};
		
		SHADER   shader;
		TEXTURES textures;
		VOLUME   volume;
		RAYCAST  ray;
		THREAD   thread;
		FPS      fps;

	}SGFLUIDVARS;

	typedef struct GRIDCPX
	{
		SGDOUBLE   u, v, w, den;
		SGBOUNDARY obstacle;
		SGDOUBLE   div;
		SGDOUBLE   p;
	} SGCUDAGRID;

	typedef struct GRIDSPL
	{
		SGDOUBLE   u, v, w, den;
	} SGHOSTGRID;

	typedef struct HOSTNODE
	{
		HOSTNODE  *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		SGBOOLEAN  bActive;
		SGINT3     n3Pos;
		GRIDSPL    *ptrGrids;
	} SGHOSTNODE;

	typedef struct DEVBUFF
	{
		GRIDCPX *ptrCenterGrids;
		GRIDSPL *ptrLeftGrids;
		GRIDSPL *ptrRightGrids;
		GRIDSPL *ptrUpGrids;
		GRIDSPL *ptrDownGrids;
		GRIDSPL *ptrFrontGrids;
		GRIDSPL *ptrBackGrids;
	} SGDEVBUFF;

}

#endif