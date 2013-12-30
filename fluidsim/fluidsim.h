/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <First>       Dec 15, 2013
* <Last>		Dec 21, 2013
* <File>        fluidsim.h
*/

#ifndef __fluid_sim_h_
#define __fluid_sim_h_

#include <vector>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <Windows.h>


#pragma region definitions

#define DELTA_TIME          1.0 // 1 second
#define DIFFUSION           0.1 // 0.1 diffusion
#define VISOCITY            0.0 // 0.0 visocity
#define INDENSITY           100 // 100 indensity

#define Grids_X             128 // 128 grids per coordination
#define Simul_Size      2097152 // 256 x 256 x 256
#define Threads_X           512 // 512 threads
#define Tile_X               16 // 16 x 16 threads as a block

#define DevListNum           11
#define dev_u                dev_list [ 0 ]
#define dev_v                dev_list [ 1 ]
#define dev_w                dev_list [ 2 ]
#define dev_u0               dev_list [ 3 ]
#define dev_v0               dev_list [ 4 ]
#define dev_w0               dev_list [ 5 ]
#define dev_den              dev_list [ 6 ]
#define dev_den0             dev_list [ 7 ]
#define dev_grid1            dev_list [ 8 ]
#define dev_grid2            dev_list [ 9 ]
#define dev_grid3            dev_list [ 10 ]

#define HostListNum          5
#define host_u               host_list [ 0 ]
#define host_v               host_list [ 1 ]
#define host_w               host_list [ 2 ]
#define host_den             host_list [ 3 ]
#define host_den0            host_list [ 4 ]

/*
  -----------------------------------------------------------------------------------------------------------
   Custom CUDA Functions
  -----------------------------------------------------------------------------------------------------------
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
  -----------------------------------------------------------------------------------------------------------
   Define something
  -----------------------------------------------------------------------------------------------------------
*/

#define eqt               ==            /* equal to */
#define and               &&            /* logical and */
#define or                ||            /* logical or */

#define gst_header        0             /* (ghost, halo) the header cell of grid */
#define sim_header        1             /* (actually) the second cell of grid */
#define gst_trailer       Grids_X - 1  /* (ghost, halo) the last cell of grid */
#define sim_trailer       Grids_X - 2  /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_trailer ) \
	if ( j >= sim_header and j <= sim_trailer ) \
	if ( k >= sim_header and k <= sim_trailer ) {

#define EndSimArea() }

#pragma endregion


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


struct param
{
	const static int nGrids_X  = Grids_X;
	const static int nSim_Size = Simul_Size;
	const static int nGridGhostHeader = gst_header;
	const static int nGridGhostTrailer = gst_trailer;
	const static int nGridSimHeader = sim_header;
	const static int nGridSimTail   = sim_trailer;
};  


namespace sge
{
	class FluidSimProc
	{
	private:
		std::vector <double*> dev_list;
		std::vector <double*> host_list;
		GLubyte *host_data;
		unsigned char *dev_data;

	public:
		FluidSimProc ( fluidsim *fluid );

		void FluidSimSolver ( fluidsim *fluid );
		void FreeResourcePtrs ( void );
		void ZeroData ( void );

	private:
		SGRUNTIMEMSG AllocateResourcePtrs ( fluidsim *fluid );
		void DensitySolver ( void );
		void VelocitySolver ( void );
		void AddSource ( void );
		void PickData ( fluidsim *fluid );
		void CopyDataToHost ( void );
		void CopyDataToDevice ( void );
	};
};


#endif