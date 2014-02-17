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
#define NODES_X                 3     // ����ڵ��ڵ�ά������ӵ�е�����
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

	/* ����ı߽����� */
	enum SGBOUNDARY
	{
		SG_BLANK,  // �հ׿ռ�
		SG_SOURCE, // Դ
		SG_WALL,   // �ϰ���
	};
	
	/* ����ĳ� */
	enum SGFIELDTYPE
	{
		SG_DENSITY_FIELD,     // �ܶ�
		SG_VELOCITY_U_FIELD,  // �ٶȳ���U�����ϵķ���
		SG_VELOCITY_V_FIELD,  // �ٶȳ���V�����ϵķ���
		SG_VELOCITY_W_FIELD,  // �ٶȳ���W�����ϵķ���
	};
	
	/* ����ڵ��λ�ñ�� */
	enum SGNODECOORD
	{
		SG_CENTER,      // λ�ڼ���ڵ��������
		SG_LEFT,        // λ�ڼ���ڵ������λ��-U����
		SG_RIGHT,       // λ�ڼ���ڵ���ҷ���λ��+U����
		SG_UP,          // λ�ڼ���ڵ���Ϸ���λ��+V����
		SG_DOWN,        // λ�ڼ���ڵ���·���λ��-V����
		SG_FRONT,       // λ�ڼ���ڵ��ǰ����λ��+W����
		SG_BACK,        // λ�ڼ���ڵ�ĺ���λ��-W����
		SG_NO_DEFINE,   // ����
	};
	
	/* ����ʽ�ļ������������ */
	enum SGJACOBI
	{
		SG_SOLVE_DENSITY,  // ������ļ�������Ϊ�����ܶȵ�����
		SG_SOLVE_VELOCITY, // ������ļ�������Ϊ�����ٶȵ�����
	};

	/* ����ģ��Ļ������ݽṹ */
	typedef struct SGSTDGRID
	{
		SGDOUBLE    u, v, w, dens;  // ˫���ȸ����������ͣ��ֱ��ʾ���ڸø����ٶȼ��ܶ�����
		SGBOUNDARY  obstacle;       // �Զ����������ͣ���ʾ���ڸø��ı߽���Ϣ
	}SGGRID;

	/* ����ڵ�����˽ṹ */
	struct SGHOSTNODE
	{
		SGHOSTNODE *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack; // ��������ڵ�����˽ṹ
		SGBOOLEAN   bActive;     // ����ֵ����ʾ�ýڵ��Ƿ񱻼���
		SGINT3      n3Pos;       // INT3����ʾ�ýڵ�Ŀռ�����
		SGSTDGRID   ptrGrids[GRIDS_X*GRIDS_X*GRIDS_X]; // �ü���ڵ��������ļ������񣬱������и����Ĭ��Ϊ64^3
	};

	/* ��ȫ����ץȡ�ֲ������ڵĽڵ㣬��GPU���в��м��� */
	struct SGCUDANODES
	{
		SGSTDGRID ptrCenter[GRIDS_X*GRIDS_X*GRIDS_X];
		SGSTDGRID ptrLeft[GRIDS_X*GRIDS_X*GRIDS_X];
		SGSTDGRID ptrRight[GRIDS_X*GRIDS_X*GRIDS_X];
		SGSTDGRID ptrUp[GRIDS_X*GRIDS_X*GRIDS_X];
		SGSTDGRID ptrDown[GRIDS_X*GRIDS_X*GRIDS_X];
		SGSTDGRID ptrFront[GRIDS_X*GRIDS_X*GRIDS_X];
		SGSTDGRID ptrBack[GRIDS_X*GRIDS_X*GRIDS_X];
	};


	/* SGCUDANODES�ĺ�����ʽ */
	struct SGSIMPLENODES
	{
		SGDOUBLE ptrCenter[GRIDS_X*GRIDS_X*GRIDS_X];
		SGDOUBLE ptrLeft[GRIDS_X*GRIDS_X*GRIDS_X];
		SGDOUBLE ptrRight[GRIDS_X*GRIDS_X*GRIDS_X];
		SGDOUBLE ptrUp[GRIDS_X*GRIDS_X*GRIDS_X];
		SGDOUBLE ptrDown[GRIDS_X*GRIDS_X*GRIDS_X];
		SGDOUBLE ptrFront[GRIDS_X*GRIDS_X*GRIDS_X];
		SGDOUBLE ptrBack[GRIDS_X*GRIDS_X*GRIDS_X];
	};

}

#endif