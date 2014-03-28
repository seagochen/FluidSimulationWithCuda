/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Nov 21, 2013
* <Last Time>     Feb 15, 2014
* <File Name>     FrameworkDynamic.h
*/

#ifndef __framework_dynamic_h_
#define __framework_dynamic_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include "MacroDefinition.h"

namespace sge
{
	typedef GLenum GLENUM;

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


	typedef class Framework_v1_0
	{
	private:
		static SGUCHAR  *DefaultTransFunc( SGVOID );
		static SGHANDLER Create1DTransFunc( GLubyte *transfer );
		static SGHANDLER Create2DCanvas( FLUIDSPARAM *fluid );
		static SGHANDLER Create2DFrameBuffer( FLUIDSPARAM *fluid );
		static SGHANDLER Create3DVolumetric( SGVOID );
		static SGHANDLER CreateVerticesBufferObj( SGVOID );

	private:
		static SGVOID SetVolumeInfoUinforms( FLUIDSPARAM *fluid );
		static SGVOID RenderingFace( GLENUM cullFace, FLUIDSPARAM *fluid );
		static SGVOID CreateShaderProg( FLUIDSPARAM *fluid );

	private:
		static SGBOOLEAN CheckHandleError( SGINT nShaderObjs, ... );
		static SGVOID CountFPS();
		static DWORD WINAPI FluidSimulationProc ( LPVOID lpParam );

	public:
		Framework_v1_0( SGMAINACTIVITY **activity, SGUINT width, SGUINT height );

	public:
		static SGVOID onMouse( SGMOUSE mouse, unsigned x, unsigned y, int degree );
		static SGVOID onKeyboard( SGKEYS keys, SGKEYSTATUS status );
		static SGVOID onDestroy( SGVOID );
		static SGVOID onDisplay( SGVOID );		
		static SGVOID onCreate( SGVOID );

	private:
		static std::string string_fmt( const std::string fmt_str, ... );
		static SGVOID SetDefaultParam( SGVOID );
	}FrameworkDynamic;
};

#endif