/**
* <Author>      Orlando Chen
* <First>       Jan 23, 2014
* <Last>		Jan 25, 2014
* <File>        DataStructures.h
*/

#ifndef __data_structures_h_
#define __data_structures_h_

#include <cuda_runtime.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

namespace sge
{
	typedef double3 SGDOUBLE3;
	typedef double4 SGDOUBLE4;
	typedef int3 SGINT3;
	typedef int4 SGINT4;

	struct FLUIDSPARAM
	{
		static struct SHADER
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
	};

	typedef struct GRIDCPX
	{
		SGDOUBLE   u, v, w, den;
		SGINT      obstacle;
		SGDOUBLE   div;
		SGDOUBLE   p;
		SGDOUBLE   temp;
	}CUDAGRID;

	typedef struct GRIDSPL
	{
		SGDOUBLE   u, v, w, den;
	}HOSTGRID;

	struct HOSTNODE
	{
		HOSTNODE  *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		SGBOOLEAN  bActive;
		SGINT3     n3Pos;
		GRIDSPL   *ptrGrids;
	};

	struct DEVBUFF
	{
		GRIDCPX  *ptrGrids;
		GRIDSPL  *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};

}

#endif