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
}

#endif