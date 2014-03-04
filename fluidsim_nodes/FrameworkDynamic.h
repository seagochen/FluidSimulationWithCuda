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

	/* ����������йصĲ����������ݽṹ */
	typedef struct FLUIDSPARAM
	{
		/* ��ɫ����Ϣ */
		struct SHADER
		{
			SGHANDLER hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag; // ��ɫ�����
			SGCHAR   *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag; // ��ɫ�������ļ�
			SGSHADER *ptrShader; // ��ɫ����
		};
		
		/* ������Ϣ */
		struct TEXTURES
		{
			SGHANDLER  hTexture1D, hTexture2D, hTexture3D, hFramebuffer; 
		};
		
		/* ����Ⱦ��Ϣ */
		struct VOLUME
		{
			SGUCHAR *ptrData; // ����Ⱦʹ�õ���ά����
			size_t   uWidth, uHeight, uDepth; // �����ݵĳ�������
		};
		
		/* ͶӰ������Ϣ */
		struct RAYCAST
		{
			SGHANDLER hCluster;   // ʹ�õĴ����ξ��
			SGINT     nAngle;     // �����ε���ת��
			SGFLOAT   fStepsize;  // ͶӰ���ߵĲ���
			size_t    uCanvasWidth, uCanvasHeight; // Framebuffer�ĳ�����
		};
		
		/* ���߳� */
		struct THREAD
		{
			DWORD   dwThreadId;  // ���߳�ID��Ϣ
			HANDLE  hThread;     // ���߳̾��
		};
		
		/* FPS��Ϣ */
		struct FPS
		{
			DWORD dwFrames;          // ��ʾ��ǰ�Ѿ����ƶ���֡
			DWORD dwCurrentTime;     // ��ʾ��ǰ��ϵͳʱ��
			DWORD dwLastUpdateTime;  // ��ʾ�ϴθ���ʱ��ϵͳʱ��
			DWORD dwElapsedTime;     // �� dwCurrentTime - dwLastUpdateTime �����ʱ����
			UINT  uFPS;              // �� dwFrames * 1000 / dwElapsedTime �õ�����FPS����
		};
		
		SHADER    shader;    // ��ɫ��
		TEXTURES  textures;  // ����
		VOLUME    volume;    // ������
		RAYCAST   ray;       // ����
		THREAD    thread;    // ���߳�
		FPS       fps;       // FPS
		SGBOOLEAN run;       // �ӳ�������״̬

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