/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Mar 11, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#include <device_launch_parameters.h>
#include <string>

typedef double const  cdouble;
typedef int const     cint;
typedef unsigned char uchar;
typedef std::string*  ptrStr;

/*********************************************************************************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

/* parameters for volume rendering */
#define STEPSIZE             0.001f
#define VOLUME_X                128

/* parameters for fluid simulation */
#define DELTATIME              0.5f
#define DIFFUSION              0.1f
#define VISOCITY               0.0f
#define DENSITY               12.5f
#define VELOCITY              15.7f
#define GRIDS_X                  64

/* hierarchy of simulation nodes */
#define NODES_X                  2
#define CURSOR_X                 1
#define GNODES_X                 2
#define HNODES_X                 2

/* CUDA device's configuration info */
#define THREADS_X             1024
#define TILE_X                  16

/* screen resolution */
#define WINDOWS_X              480
#define CANVAS_X               480

/* etc */
#define TPBUFFER_X            1024

/* macro definition of grid types */
#define MACRO_DENSITY            0
#define MACRO_VELOCITY_U         1
#define MACRO_VELOCITY_V         2
#define MACRO_VELOCITY_W         3
#define MACRO_SIMPLE             4

/* macro definition of boundary condition */
#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_SOURCE     1
#define MACRO_BOUNDARY_OBSTACLE 100

/* macro definition of node's position */
#define MACRO_CENTER              0
#define MACRO_LEFT                1
#define MACRO_RIGHT               2
#define MACRO_UP                  3
#define MACRO_DOWN                4
#define MACRO_FRONT               5
#define MACRO_BACK                6

/* True and False */
#define MACRO_FALSE               0
#define MACRO_TRUE                1

/* switch */
#define TESTING_MODE_SWITCH       0 /* switch: close(0) open(1) */
#define TESTING_MODE              0 /* velocity: default-up(0) down(1) left(2) right(3) front(4) back(5) */

/*********************************************************************************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

#define cudaIndex2D(i,j,elements_x) ((j)*(elements_x)+(i))
#define cudaIndex3D(i,j,k,elements_x) ((k)*elements_x*elements_x+(j)*elements_x+(i))
#define Index(i,j,k) cudaIndex3D(i,j,k,GRIDS_X)

#define gst_header                0  /* (ghost, halo) the header cell of grid */
#define sim_header                1  /* (actually) the second cell of grid */
#define gst_tailer               63  /* (ghost, halo) the last cell of grid */
#define sim_tailer               62  /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_tailer ) \
	if ( j >= sim_header and j <= sim_tailer ) \
	if ( k >= sim_header and k <= sim_tailer ) {

#define EndSimArea() }

#define dev_buffers_num                   35
#define dev_den              dev_buffers[ 0 ]
#define dev_den0             dev_buffers[ 1 ]
#define dev_u                dev_buffers[ 2 ]
#define dev_u0               dev_buffers[ 3 ]
#define dev_v                dev_buffers[ 4 ]
#define dev_v0               dev_buffers[ 5 ]
#define dev_w                dev_buffers[ 6 ]
#define dev_w0               dev_buffers[ 7 ]
#define dev_div              dev_buffers[ 8 ]
#define dev_p                dev_buffers[ 9 ]
#define dev_obs              dev_buffers[ 10 ]

#define dens_C               dev_buffers[ 0 ]
#define dens_L               dev_buffers[ 11 ]
#define dens_R               dev_buffers[ 12 ]
#define dens_U               dev_buffers[ 13 ]
#define dens_D               dev_buffers[ 14 ]
#define dens_F               dev_buffers[ 15 ]
#define dens_B               dev_buffers[ 16 ]

#define velu_C               dev_buffers[ 2 ]
#define velu_L               dev_buffers[ 17 ] 
#define velu_R               dev_buffers[ 18 ]
#define velu_U               dev_buffers[ 19 ]
#define velu_D               dev_buffers[ 20 ]
#define velu_F               dev_buffers[ 21 ]
#define velu_B               dev_buffers[ 22 ]

#define velv_C               dev_buffers[ 4 ]
#define velv_L               dev_buffers[ 23 ]
#define velv_R               dev_buffers[ 24 ]
#define velv_U               dev_buffers[ 25 ]
#define velv_D               dev_buffers[ 26 ]
#define velv_F               dev_buffers[ 27 ]
#define velv_B               dev_buffers[ 28 ]

#define velw_C               dev_buffers[ 6 ]
#define velw_L               dev_buffers[ 29 ]
#define velw_R               dev_buffers[ 30 ]
#define velw_U               dev_buffers[ 31 ]
#define velw_D               dev_buffers[ 32 ]
#define velw_F               dev_buffers[ 33 ]
#define velw_B               dev_buffers[ 34 ]

/*********************************************************************************************************/
/*********************************************************************************************************/
/*********************************************************************************************************/

#define cudaTrans2DTo3D(i,j,k,elements_x) \
	k = cudaIndex2D(i,j,(elements_x)) / ((elements_x)*(elements_x)); \
	i = i % (elements_x); \
	j = j % (elements_x); \

#define cudaDeviceDim1D() \
	blockDim.x = TPBUFFER_X; \
	blockDim.y = 1; \
	gridDim.x  = 1; \
	gridDim.y  = 1; \

#define cudaDeviceDim2D() \
	blockDim.x = TILE_X; \
	blockDim.y = TILE_X; \
	gridDim.x  = GRIDS_X / TILE_X; \
	gridDim.y  = GRIDS_X / TILE_X; \

#define cudaDeviceDim3D() \
	blockDim.x = (GRIDS_X / TILE_X); \
	blockDim.y = (THREADS_X / TILE_X); \
	gridDim.x  = (GRIDS_X / blockDim.x); \
	gridDim.y  = (GRIDS_X * GRIDS_X * GRIDS_X) / (blockDim.x * blockDim.y * (GRIDS_X / blockDim.x)); \

#define __device_func__ <<<gridDim,blockDim>>>

#define GetIndex1D() \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \

#define GetIndex2D() \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \

#define GetIndex3D()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D ( i, j, k, GRIDS_X ); \

#endif