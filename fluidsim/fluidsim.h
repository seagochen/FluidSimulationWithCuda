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

#define DELTA_TIME           0.01f/* -------------------------------------- 0.1 second */
#define DIFFUSION            0.1f /* -------------------------------------- diffusion */
#define VISCOSITY            0.0f /* -------------------------------------- viscosity */
#define FORCE                5.0f /* -------------------------------------- external force */
#define SOURCE               150  /* -------------------------------------- indensity */
#define SIM_SIZE             Grids_X * Grids_X * Grids_X /* ----------- CFD dimension of grid */
#define Grids_X              64   /* ----------------------------------- grids number on each dimension */
#define SimArea_X            62   /* ----------------------------------- number of grids without ghost cells */
#define Threads_X            512  /* ----------------------------------- number of threads enabled */
#define Tile_X               16   /* ----------------------------------- ties 16x16 gpu threads as a block */

#define DevListNum    10
#define dev_u         dev_list [ 0 ] /* -------------------------- u of U, on CUDA */
#define dev_v         dev_list [ 1 ] /* -------------------------- v of U, on CUDA */
#define dev_w         dev_list [ 2 ] /* -------------------------- w of U, on CUDA */
#define dev_u0        dev_list [ 3 ] /* -------------------------- original u of U, on CUDA */
#define dev_v0        dev_list [ 4 ] /* -------------------------- original v of U, on CUDA */
#define dev_w0        dev_list [ 5 ] /* -------------------------- original w of U, on CUDA */
#define dev_den       dev_list [ 6 ] /* -------------------------- density, on CUDA */
#define dev_den0      dev_list [ 7 ] /* -------------------------- original density, on CUDA */
#define dev_grid      dev_list [ 8 ] /* -------------------------- temporary grid, on CUDA */
#define dev_grid0     dev_list [ 9 ] /* -------------------------- original temporary grid, on CUDA */

#define HostListNum   8
#define host_u        host_list [ 0 ] /* ------------------------- component u of flow U on x-axis (left to right) */
#define host_v        host_list [ 1 ] /* ------------------------- component v of flow U on y-axis (up to down) */
#define host_w        host_list [ 2 ] /* ------------------------- component w of flow U on z-axis (near to far) */
#define host_u0       host_list [ 3 ] /* ------------------------- original u */
#define host_v0       host_list [ 4 ] /* ------------------------- original v */
#define host_w0       host_list [ 5 ] /* ------------------------- original w */
#define host_den      host_list [ 6 ] /* ------------------------- scalar field of density */
#define host_den0     host_list [ 7 ] /* ------------------------- original density */

/*
  -----------------------------------------------------------------------------------------------------------
   Custom CUDA Functions
  -----------------------------------------------------------------------------------------------------------
*/

#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			std::cout << "in file: " << __FILE__ << ", at line: " << __LINE__ << std::endl; \
			std::cout << "error: " << cudaGetErrorString(__err) << std::endl; \
			std::cout << msg << std::endl; } \
	} while(0); \

#define cudaDevice(gridDim, blockDim) <<<gridDim, blockDim>>>

#define cudaIndex2D(i, j, elements_x) ((j) * (elements_x) + (i))

#define cudaTrans2DTo3D(i, j, k, elements_x) { \
	k = cudaIndex2D(i, j, elements_x) / ((elements_x) * (elements_x)) ; \
	i = i % elements_x; \
	j = j % elements_x; \
	}

#define cudaIndex3D(i, j, k, elements_x) ((k) * elements_x * elements_x + (j) * elements_x + (i))

/*
  -----------------------------------------------------------------------------------------------------------
   Function Definitions
  -----------------------------------------------------------------------------------------------------------
*/

#define Index(i, j, k)      cudaIndex3D(i, j, k, Grids_X)

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = Tile_X; \
	blockDim.y = Tile_X; \
	gridDim.x  = Grids_X / Tile_X; \
	gridDim.y  = Grids_X / Tile_X;

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (Grids_X / Tile_X); \
	blockDim.y = (Threads_X / Tile_X); \
	gridDim.x  = (Grids_X / blockDim.x); \
	gridDim.y  = (Grids_X * Grids_X * Grids_X) / (blockDim.x * blockDim.y * (Grids_X / blockDim.x));

#define GetIndex()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D ( i, j, k, Grids_X );

/*
  -----------------------------------------------------------------------------------------------------------
   Define something
  -----------------------------------------------------------------------------------------------------------
*/

#define eqt              ==            /* equal to */
#define and              &&            /* logical and */
#define or               ||            /* logical or */

#define gst_header       0             /* (ghost, halo) the header cell of grid */
#define sim_header       1             /* (actually) the second cell of grid */
#define gst_trailer       Grids_X - 1   /* (ghost, halo) the last cell of grid */
#define sim_trailer       Grids_X - 2   /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_trailer ) \
	if ( j >= sim_header and j <= sim_trailer ) \
	if ( k >= sim_header and k <= sim_trailer ) {

#define EndSimArea() }

#pragma endregion


#pragma region structures

struct fluidsim
{
	/// Handles for shader objects, and programs ///
	GLuint hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
	/// Handles for vertices buffer object ///
	GLuint hCluster;
	/// Handles for textures, and frame buffer ///
	GLuint hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
	/// Shader Helper ///
	sge::Shader *ptrShader;
	/// 3-D volumetric data ///
	GLubyte *ptrData;
	/// 3-D volumetric data information ///
	GLint nVolWidth, nVolHeight, nVolDepth;
	/// Source files for GLSL ///
	char *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
	/// Screen parameters ///	
	GLint nAngle;
	bool  bFullScreen;
	/// Canvas information ///
	GLint nCanvasWidth, nCanvasHeight;
	/// Stride of ray casting ///
	GLfloat  fStepsize;
	/// Thread ID and handle ///
	DWORD   dwThreadId;
	HANDLE  hThread;
};

struct param
{
	const static int nGrids_X  = Grids_X;
	const static int nSim_Size = SIM_SIZE;
	const static int nGridGhostHeader = gst_header;
	const static int nGridGhostTrailer = gst_trailer;
	const static int nGridSimHeader = sim_header;
	const static int nGridSimTail   = sim_trailer;
};  

#pragma endregion


namespace sge
{
	class FluidSim
	{
	private:
		std::vector <float*> dev_list;         /* ------------------------ a vector for storing device ptr */
		std::vector <float*> host_list;        /* ------------------------ a vector for storing host ptr */

	public:
		FluidSim ( fluidsim *fluid );

		void FluidSimSolver ( fluidsim *fluid );
		void FreeResourcePtrs ( void );
		void ZeroData ( void );

	private:
		SGRUNTIMEMSG AllocateResourcePtrs ( void );

	private:
		GLubyte *data;
		unsigned char* dev_data;
		bool first;
	};
};


#endif