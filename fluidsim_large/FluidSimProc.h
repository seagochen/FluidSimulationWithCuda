/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 20, 2014
* <File Name>     FluidSimProc.h
*/


#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "FunctionHelper.h"
#include "FrameworkDynamic.h"

using std::vector;

namespace sge
{
	class FluidSimProc
	{
	private:
		struct SimNode
		{
			SGBOOLEAN updated;
			SGINT3 nodeIX;
			SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		};

	private:
		/* Level-0 GPU */
		vector <double*> dev_obstacle;

		/* Level-0 GPU */
		vector <double*> dev_buffers;

		/* Level-1 GPU */
		vector <double*> dev_density_s;
		vector <double*> dev_velocity_u_s;
		vector <double*> dev_velocity_v_s;
		vector <double*> dev_velocity_w_s;
		
		/* Level-1 GPU */
		vector <double*> dev_density_t;
		vector <double*> dev_velocity_u_t;
		vector <double*> dev_velocity_v_t;
		vector <double*> dev_velocity_w_t;

		/* Level-0 Host */
		vector <double*> host_density;
		vector <double*> host_velocity_u;
		vector <double*> host_velocity_v;
		vector <double*> host_velocity_w;
		vector <double*> host_obstacle;

		vector <SimNode*> gpu_node, host_node;

		/* 可视化 */
		SGUCHAR *dev_visual, *host_visual;

		/* 临时数据 */
		double *dev_dtpbuf, *host_dtpbuf;
		int    *dev_ntpbuf, *host_ntpbuf;

	private:
		/* cursor */
		SGINT3 m_cursor;

		/* CUDA */
		dim3 gridDim, blockDim;

	private:
		FunctionHelper helper;

	private:
		/* node and volumetric size */
		size_t m_node_size;
		size_t m_volm_size;

		/* 程序窗口标题 */
		std::string m_sz_title;

		/* 与AddSource时有关的变量 */
		int increase_times, decrease_times;

	public:
		FluidSimProc( FLUIDSPARAM *fluid );

		/* fluid simulation processing function */
		void FluidSimSolver( FLUIDSPARAM *fluid );

		/* when program existed, release resource */
		void FreeResource( void );

		/* zero the buffers for fluid simulation */
		void ZeroBuffers( void );

		/* 上传内存节点数据 */
		void UploadNodes( void );

		/* 下载缓存节点数据 */
		void DownloadNodes( void );

		/* 获得窗口标题，版本号，采用的技术, etc. */
		ptrStr GetTitleBar( void );

		/* 打印当前的节点信息 */
		void PrintMSG( void );

	private:
		/* flood buffer for multiple nodes */
		void InteractNodes( int i, int j, int k );

		/* initialize FPS and etc. */
		void InitParams( FLUIDSPARAM *fluid );

		/* copy host data to CUDA device */
		void LoadNode( int i, int j, int k );

		/* retrieve data back to host */
		void SaveNode( int i, int j, int k );
			
		/* retrieve the density back and load into volumetric data for rendering */
		void Finally( FLUIDSPARAM *fluid );
		
		/* create simulation nodes' topological structure */
		void CreateTopology( void );

		/* allocate resource */
		bool AllocateResource( FLUIDSPARAM *fluid );

		/* solving density */
		void DensitySolver( void );

		/* add source */
		void AddSource( void );

		/* initialize boundary condition */
		void InitBoundary( int i, int j, int k );

		/* solving velocity */
		void VelocitySolver( void );
	};
};

#endif