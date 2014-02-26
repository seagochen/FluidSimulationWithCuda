/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Feb 17, 2014
* <File Name>     DataStructures.h
*/

#ifndef __data_and_structures_h_
#define __data_and_structures_h_

#include <SGE\SGUtils.h>
#include <device_launch_parameters.h>

namespace sge
{
	typedef double3 SGDOUBLE3;
	typedef double4 SGDOUBLE4;
	typedef int3    SGINT3;
	typedef int4    SGINT4;

	/* 与流体计算有关的参数包的数据结构 */
	typedef struct FLUIDSPARAM
	{
		/* 着色器信息 */
		struct SHADER
		{
			SGHANDLER hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag; // 着色器句柄
			SGCHAR   *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag; // 着色器代码文件
			SGSHADER *ptrShader; // 着色程序
		};
		
		/* 纹理信息 */
		struct TEXTURES
		{
			SGHANDLER  hTexture1D, hTexture2D, hTexture3D, hFramebuffer; 
		};
		
		/* 体渲染信息 */
		struct VOLUME
		{
			SGUCHAR *ptrData; // 体渲染使用的三维数据
			size_t   uWidth, uHeight, uDepth; // 体数据的长、宽、高
		};
		
		/* 投影射线信息 */
		struct RAYCAST
		{
			SGHANDLER hCluster;   // 使用的代理几何句柄
			SGINT     nAngle;     // 代理几何的旋转角
			SGFLOAT   fStepsize;  // 投影射线的步长
			size_t    uCanvasWidth, uCanvasHeight; // Framebuffer的长、宽
		};
		
		/* 多线程 */
		struct THREAD
		{
			DWORD   dwThreadId;  // 子线程ID信息
			HANDLE  hThread;     // 子线程句柄
		};
		
		/* FPS信息 */
		struct FPS
		{
			DWORD dwFrames;          // 表示当前已经绘制多少帧
			DWORD dwCurrentTime;     // 表示当前的系统时间
			DWORD dwLastUpdateTime;  // 表示上次跟新时的系统时间
			DWORD dwElapsedTime;     // 由 dwCurrentTime - dwLastUpdateTime 计算出时间间隔
			UINT  uFPS;              // 由 dwFrames * 1000 / dwElapsedTime 得到最终FPS数据
		};
		
		SHADER    shader;    // 着色器
		TEXTURES  textures;  // 纹理
		VOLUME    volume;    // 体数据
		RAYCAST   ray;       // 射线
		THREAD    thread;    // 子线程
		FPS       fps;       // FPS
		SGBOOLEAN run;       // 子程序运行状态

	} SGFLUIDVARS;
}

#endif