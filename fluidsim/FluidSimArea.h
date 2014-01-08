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

#define DELTA_TIME          0.5f 
#define DIFFUSION           0.1f
#define VISOCITY            0.0f 
#define VOLUME              15

#define Threads_X           512
#define Grids_X             128
#define Tile_X               16


#pragma region definitions

#define Sim_Size     Grids_X*Grids_X*Grids_X

#define DevListNum           12
#define dev_u                dev_list [ 0 ]
#define dev_v                dev_list [ 1 ]
#define dev_w                dev_list [ 2 ]
#define dev_den              dev_list [ 3 ]
#define dev_div              dev_list [ 4 ]
#define dev_p                dev_list [ 5 ]
#define dev_0                dev_list [ 6 ]
#define dev_1                dev_list [ 7 ]
#define dev_2                dev_list [ 8 ]
#define dev_3                dev_list [ 9 ]
#define dev_4                dev_list [ 10 ]
#define dev_5                dev_list [ 11 ]

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


namespace sge
{
	struct fluidsim
	{
		static struct shader
		{
			/// Handles for shader objects, and programs ///
			GLuint hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
			/// Shader Helper ///
			sge::Shader *ptrShader;
			/// Source files for GLSL ///
			char *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
		};
		
		struct textures
		{
			/// Handles for textures, and frame buffer ///
			GLuint hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
		};
		
		struct volume
		{
			/// 3-D volumetric data ///
			GLubyte *ptrData;
			/// 3-D volumetric data information ///
			GLint nVolWidth, nVolHeight, nVolDepth;
		};
		
		struct drawing
		{
			/// Handles for vertices buffer object ///
			GLuint hCluster;
			/// Screen parameters ///
			GLint nAngle;
			bool  bFullScreen, bContinue;
			/// Canvas information ///
			GLint nCanvasWidth, nCanvasHeight;
			/// Stride of ray casting ///
			GLfloat  fStepsize;
		};
		
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
			UINT  FPS;
		};
		
		shader   shader;
		textures textures;
		volume   volume;
		drawing  drawing;
		thread   thread;
		fps      fps;
	};
	
	struct constparam
	{
		const static int nGrids_X          = Grids_X;
		const static int nSim_Size         = Sim_Size;
		const static int nGridGhostHeader  = gst_header;
		const static int nGridGhostTrailer = gst_trailer;
		const static int nGridSimHeader    = sim_header;
		const static int nGridSimTail      = sim_trailer;
		const static int nGlobalI          = 1;
		const static int nGlobalJ          = 1;
		const static int nGlobalK          = 1;
	};

	struct dataset 
	{
		uchar *host_visual, *dev_visual;
		double *big_den, *big_u, *big_v, *big_w;
	};
	
	class FluidSimProc
	{
	private:
		std::vector <double*> dev_list;
		std::vector <double*> host_list;
		dataset data;
		uchar *host_data, *dev_data;

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