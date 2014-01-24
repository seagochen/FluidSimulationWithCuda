/**
* <Author>      Orlando Chen
* <First>       Jan 23, 2014
* <Last>		Jan 25, 2014
* <File>        DataStructures.h
*/

#ifndef __data_structures_h_
#define __data_structures_h_

#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

namespace sge
{
	struct fluidsim
	{
		static struct shader
		{
			SGHANDLER hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
			char     *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
			SGSHADER *ptrShader;
		};
		
		struct textures
		{
			SGHANDLER  hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
		};
		
		struct volume
		{
			SGUCHAR *ptrData;
			size_t   uWidth, uHeight, uDepth;
		};
		
		typedef struct raycasting
		{
			SGHANDLER hCluster;
			SGINT     nAngle;
			SGBOOLEAN bRun;
			SGFLOAT   fStepsize;
			size_t    uCanvasWidth, uCanvasHeight;
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
		
		shader   shader;
		textures textures;
		volume   volume;
		rayc     ray;
		thread   thread;
		fps      fps;
	};

#endif