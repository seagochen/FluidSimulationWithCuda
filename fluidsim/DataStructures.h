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

#define DELTATIME              0.5f   // ����0.5sΪһ��������delta time
#define STEPSIZE               0.001f // ����0.001Ϊһ���������
#define DIFFUSION              0.1f   // diffusion�Ķ���ֵΪ0.1
#define VISOCITY               0.0f   // visocity�Ķ���ֵΪ0.1
#define SOURCE_DENSITY         30     // Ϊ������������ӵ�density��Ũ��
#define SOURCE_VELOCITY        15     // Ϊ������������ӵ�velocity����

#define GRIDS_X                64     // ���������ڵ�ά������ӵ�е�����
#define NODES_X                 2     // ����ڵ��ڵ�ά������ӵ�е�����
#define VOLUME_X   GRIDS_X*NODES_X    // ��ά�������ڵ�ά���ϵĳ���
#define THREADS_X             1024    // ����CUDA���߳�����
#define TILE_X                 16     // ��16x16��GPU-threads������Ϊһ��block
#define WINDOWS_X              600    // Windows application's size
#define CANVAS_X               600    // canvas's size
#define TPBUFFER_X             1024   // Ϊ�˿���ֲ�Զ���������ʱ���ݻ��棬�������shared memories

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