/**
* <Author>      Orlando Chen
* <First>       Dec 15, 2013
* <Last>		Jan 07, 2014
* <File>        FluidSimArea.h
*/

#ifndef __fluid_sim_h_
#define __fluid_sim_h_

#include <vector>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <Windows.h>

#define DELTA_TIME          0.4f 
#define DIFFUSION           0.1f
#define VISOCITY            0.0f 
#define VOLUME              15

#define Threads_X           512
#define Grids_X             64
#define Area_X          Grids_X
#define Tile_X               16


#pragma region get index, host & device list, simulation area control

#define Sim_Size     Grids_X*Grids_X*Grids_X

#define DevListNum           17
#define dev_u                dev_list [ 0 ]
#define dev_v                dev_list [ 1 ]
#define dev_w                dev_list [ 2 ]
#define dev_den              dev_list [ 3 ]
#define dev_obs              dev_list [ 4 ]
#define dev_div              dev_list [ 5 ]
#define dev_p                dev_list [ 6 ]
#define dev_u0               dev_list [ 7 ]
#define dev_v0               dev_list [ 8 ]
#define dev_w0               dev_list [ 9 ]
#define dev_den0             dev_list [ 10 ]
#define dev_0                dev_list [ 11 ]
#define dev_1                dev_list [ 12 ]
#define dev_2                dev_list [ 13 ]
#define dev_3                dev_list [ 14 ]
#define dev_4                dev_list [ 15 ]
#define dev_5                dev_list [ 16 ]

#define HostListNum          4
#define host_u               host_list [ 0 ]
#define host_v               host_list [ 1 ]
#define host_w               host_list [ 2 ]
#define host_den             host_list [ 3 ]

/*
  -------------------------------------------------------------------------------------------------------
   Custom CUDA Functions
  -------------------------------------------------------------------------------------------------------
*/

#include <stdio.h>
#include <cuda_runtime.h>

inline void cudaCheckErrors ( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};

#define cudaDevice(gridDim,blockDim) <<<gridDim,blockDim>>>

#define cudaIndex2D(i,j,elements_x) ((j)*(elements_x)+(i))

#define cudaIndex3D(i,j,k,elements_x) ((k)*elements_x*elements_x+(j)*elements_x+(i))

#define Index(i,j,k) cudaIndex3D(i,j,k,Grids_X)

#define cudaTrans2DTo3D(i,j,k,elements_x) \
	k = cudaIndex2D(i,j,(elements_x)) / ((elements_x)*(elements_x)); \
	i = i % (elements_x); \
	j = j % (elements_x); \

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = Tile_X; \
	blockDim.y = Tile_X; \
	gridDim.x  = Grids_X / Tile_X; \
	gridDim.y  = Grids_X / Tile_X; \

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (Grids_X / Tile_X); \
	blockDim.y = (Threads_X / Tile_X); \
	gridDim.x  = (Grids_X / blockDim.x); \
	gridDim.y  = (Grids_X * Grids_X * Grids_X) / (blockDim.x * blockDim.y * (Grids_X / blockDim.x)); \

#define GetIndex()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D ( i, j, k, Grids_X ); \

/*
  -------------------------------------------------------------------------------------------------------
   Define something
  -------------------------------------------------------------------------------------------------------
*/

#define eqt               ==
#define and               &&
#define or                ||

#define gst_header        0              /* (ghost, halo) the header cell of grid */
#define sim_header        1              /* (actually) the second cell of grid */
#define gst_trailer       Grids_X - 1    /* (ghost, halo) the last cell of grid */
#define sim_trailer       Grids_X - 2    /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_trailer ) \
	if ( j >= sim_header and j <= sim_trailer ) \
	if ( k >= sim_header and k <= sim_trailer ) {

#define EndSimArea() }

#pragma endregion


typedef GLuint handler;

/*
  -------------------------------------------------------------------------------------------------------
   Dataset
  -------------------------------------------------------------------------------------------------------
*/

namespace sge
{
	/// structure of fluidsim ///
	struct fluidsim
	{
#pragma region definition of structures, such as shader, textures, volume, fps and etc.
		static struct shader
		{
			handler hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
			char   *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
			Shader *ptrShader;
		};
		
		struct textures
		{
			handler  hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
		};
		
		struct volume
		{
			uchar  *ptrData;
			size_t  uWidth, uHeight, uDepth;
		};

		struct area
		{
			double *ptrVelU, *ptrVelV, *ptrVelW;
			double *ptrDens;
			size_t  uWidth, uHeight, uDepth;
		};
		
		typedef struct raycasting
		{
			handler hCluster;
			int     nAngle;
			boolean bRun;
			GLfloat fStepsize;
			size_t  uCanvasWidth, uCanvasHeight;
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
#pragma endregion
		
		shader   shader;
		textures textures;
		volume   volume;
		area     area;
		rayc     ray;
		thread   thread;
		fps      fps;
	};

	/// fluid simulation processor ///
	class FluidSimProc
	{
		struct node
		{
			node    *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
			boolean  bActive;
			size_t   uWidth, uHeight, uDepth;
		};

	private:
		std::vector <double*> dev_list;
		std::vector <double*> host_list;
		std::vector <node>    node_list;
		uchar  *host_visual, *dev_visual;
		
		int nOffi, nOffj, nOffk; // offset, default value is 0

	public:
		FluidSimProc ( fluidsim *fluid );

		void FluidSimSolver ( fluidsim *fluid );
		void FreeResourcePtrs ( void );
		void ZeroData ( void );

	private:
		SGRUNTIMEMSG AllocateResourcePtrs ( fluidsim *fluid );
		void DensitySolver ( void );
		void VelocitySolver ( void );
		void PickData ( fluidsim *fluid );
		void CopyDataToHost ( void );
		void CopyDataToDevice ( void );
	};
};

#endif