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

#define DELTATIME              0.5f   // 定义0.5s为一个步长的delta time
#define STEPSIZE               0.001f // 定义0.001为一个步长深度
#define DIFFUSION              0.1f   // diffusion的定义值为0.1
#define VISOCITY               0.0f   // visocity的定义值为0.1
#define SOURCE_DENSITY         30     // 为计算网格中添加的density的浓度
#define SOURCE_VELOCITY        15     // 为计算网格中添加的velocity的量

#define GRIDS_X                64     // 计算网格在单维度上所拥有的数量
#define NODES_X                 3     // 计算节点在单维度上所拥有的数量
#define VOLUME_X   GRIDS_X*NODES_X    // 三维体数据在单维度上的长度
#define THREADS_X             1024    // 定义CUDA的线程数量
#define TILE_X                 16     // 将16x16的GPU-threads捆绑打包为一个block
#define WINDOWS_X              600    // Windows application's size
#define CANVAS_X               600    // canvas's size
#define TPBUFFER_X             1024   // 为了可移植性而创建的临时数据缓存，用于替代shared memories

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

	/* 定义的边界条件 */
	enum SGBOUNDARY
	{
		SG_BLANK,  // 空白空间
		SG_SOURCE, // 源
		SG_WALL,   // 障碍物
	};
	
	/* 定义的场 */
	enum SGFIELDTYPE
	{
		SG_DENSITY_FIELD,     // 密度
		SG_VELOCITY_U_FIELD,  // 速度场在U方向上的分量
		SG_VELOCITY_V_FIELD,  // 速度场在V方向上的分量
		SG_VELOCITY_W_FIELD,  // 速度场在W方向上的分量
	};
	
	/* 计算节点的位置标记 */
	enum SGNODECOORD
	{
		SG_CENTER,      // 位于计算节点的正中央
		SG_LEFT,        // 位于计算节点的左方向，位于-U方向
		SG_RIGHT,       // 位于计算节点的右方向，位于+U方向
		SG_UP,          // 位于计算节点的上方向，位于+V方向
		SG_DOWN,        // 位于计算节点的下方向，位于-V方向
		SG_FRONT,       // 位于计算节点的前方向，位于+W方向
		SG_BACK,        // 位于计算节点的后方向，位于-W方向
		SG_NO_DEFINE,   // 其他
	};
	
	/* 行列式的计算的数据类型 */
	enum SGJACOBI
	{
		SG_SOLVE_DENSITY,  // 所处理的计算数据为关于密度的数据
		SG_SOLVE_VELOCITY, // 所处理的计算数据为关于速度的数据
	};

	/* 流体模拟的基本数据结构 */
	typedef struct SGSTDGRID
	{
		SGDOUBLE    u, v, w, dens;  // 双精度浮点数据类型，分别表示处于该格点的速度及密度数据
		SGBOUNDARY  obstacle;       // 自定义数据类型，表示处于该格点的边界信息
	}SGGRID;

	/* 计算节点的拓扑结构 */
	struct SGHOSTNODE
	{
		SGHOSTNODE *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack; // 链表，计算节点的拓扑结构
		SGBOOLEAN   bActive;     // 布尔值，表示该节点是否被激活
		SGINT3      n3Pos;       // INT3，表示该节点的空间坐标
		SGSTDGRID   ptrGrids[GRIDS_X*GRIDS_X*GRIDS_X]; // 该计算节点所包含的计算网格，本程序中格点数默认为64^3
	};

	/* 从全局中抓取局部的相邻的节点，在GPU进行并行计算 */
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


	/* SGCUDANODES的簡化形式 */
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