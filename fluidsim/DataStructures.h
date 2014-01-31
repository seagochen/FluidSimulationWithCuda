/**
* <Author>      Orlando Chen
* <First>       Jan 23, 2014
* <Last>		Jan 31, 2014
* <File>        DataStructures.h
*/

#ifndef __data_and_structures_h_
#define __data_and_structures_h_

#include <SGE\SGUtils.h>
#include <device_launch_parameters.h>

#define DELTATIME           0.5f
#define STEPSIZE            0.001f
#define DIFFUSION           0.1f
#define VISOCITY            0.0f 
#define SOURCE_DENSITY      30
#define SOURCE_VELOCITY     15

#define BASIC_X             64     // standard grids on one dimension is 64
#define GRIDS_X             65     // extend grids on one dimension is 65
#define NODES_X             3      // standard nodes on one dimension is 3
#define VOLUME_X            192    // rendering box on one dimension is 3x64=192
#define THREADS_X           512    // standard GPU-threads per sm
#define TILE_X              16     // 16x16 GPU-threads as a block
#define WINDOWS_X           600    // Windows application's size
#define CANVAS_X            600    // canvas's size
#define TPBUFFER_X          1024   // size of GPU temporary buffer is 1024, as shared memory

namespace sge
{
	typedef double3 SGDOUBLE3;
	typedef double4 SGDOUBLE4;
	typedef int3    SGINT3;
	typedef int4    SGINT4;

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

	} SGFLUIDVARS;

	enum SGBOUNDARY
	{
		SG_BD_SOURCE,
		SG_BD_BLANK,
		SG_BD_WALL,
	};
	
	enum SGFIELDTYPE
	{
		SG_DENSITY_FIELD,
		SG_VELOCITY_U_FIELD,
		SG_VELOCITY_V_FIELD,
		SG_VELOCITY_W_FIELD,
	//	SG_DIVERGENCE_FIELD,
	//	SG_PRESSURE_FIELD,
	};
	
	enum SGUSINGNODE
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
	
	enum SGJACOBI
	{
		SG_SOLVE_DENSITY,
		SG_SOLVE_VELOCITY,
	};

	/* standard format of grids */
	typedef struct SGSTDGRID
	{
		SGDOUBLE    u, v, w, dens;
		SGBOUNDARY  obstacle;
	}SGGRID;

	/* structure of host node */
	struct SGHOSTNODE
	{
		SGHOSTNODE *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		SGBOOLEAN   bActive;
		SGINT3      n3Pos;
		SGSTDGRID  *prtGrids;
	};

	/*  */
	struct SGCUDABUFF
	{
		SGSTDGRID *ptrCenter, *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};
}

#endif