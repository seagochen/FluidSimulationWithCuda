/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Mar 03, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#include <device_launch_parameters.h>

namespace sge
{
	typedef double3 SGDOUBLE3;
	typedef double4 SGDOUBLE4;
	typedef int3    SGINT3;
	typedef int4    SGINT4;
}

#define DELTATIME              0.5f   // 定义0.5s为一个步长的delta time
#define STEPSIZE              0.001f  // 定义0.001为一个步长深度
#define DIFFUSION              0.1f   // diffusion的定义值为0.1
#define VISOCITY               0.0f   // visocity的定义值为0.1
#define SOURCE_DENSITY         100    // 为计算网格中添加的density的浓度
#define SOURCE_VELOCITY        100    // 为计算网格中添加的velocity的量

#define GRIDS_X                 64    // 计算网格在单维度上所拥有的数量
#define NODES_X                  3    // 计算节点在单维度上所拥有的数量
#define VOLUME_X   GRIDS_X*NODES_X    // 三维体数据在单维度上的长度
#define THREADS_X             1024    // 定义CUDA的线程数量
#define TILE_X                  16    // 将16x16的GPU-threads捆绑打包为一个block
#define WINDOWS_X              600    // Windows application's size
#define CANVAS_X               600    // canvas's size
#define TPBUFFER_X            1024    // 为了可移植性而创建的临时数据缓存，用于替代shared memories

#define MACRO_DENSITY            0
#define MACRO_VELOCITY_U         1
#define MACRO_VELOCITY_V         2
#define MACRO_VELOCITY_W         3
#define MACRO_SIMPLE             4

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_SOURCE     1
#define MACRO_BOUNDARY_OBSTACLE 100

#define MACRO_CENTER              0
#define MACRO_LEFT                1
#define MACRO_RIGHT               2
#define MACRO_UP                  3
#define MACRO_DOWN                4
#define MACRO_FRONT               5
#define MACRO_BACK                6

#define TESTING_MODE_SWITCH       0 /* switch: close(0) open(1) */
#define TESTING_MODE              0 /* velocity: default-up(0) down(1) left(2) right(3) front(4) back(5) */

#define cudaIndex2D(i,j,elements_x) ((j)*(elements_x)+(i))
#define cudaIndex3D(i,j,k,elements_x) ((k)*elements_x*elements_x+(j)*elements_x+(i))
#define Index(i,j,k) cudaIndex3D(i,j,k,GRIDS_X)

#define gst_header                0  /* (ghost, halo) the header cell of grid */
#define sim_header                1  /* (actually) the second cell of grid */
#define gst_tailer      GRIDS_X - 1  /* (ghost, halo) the last cell of grid */
#define sim_tailer      GRIDS_X - 2  /* (actually) the second last cell of grid */

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


#define cudaTrans2DTo3D(i,j,k,elements_x) \
	k = cudaIndex2D(i,j,(elements_x)) / ((elements_x)*(elements_x)); \
	i = i % (elements_x); \
	j = j % (elements_x); \

#define cudaDeviceDim1D() \
	dim3 blockDim, gridDim; \
	blockDim.x = TPBUFFER_X; \
	blockDim.y = 1; \
	gridDim.x  = 1; \
	gridDim.y  = 1; \

#define cudaDeviceDim2D() \
	dim3 blockDim, gridDim; \
	blockDim.x = TILE_X; \
	blockDim.y = TILE_X; \
	gridDim.x  = GRIDS_X / TILE_X; \
	gridDim.y  = GRIDS_X / TILE_X; \

#define cudaDeviceDim3D() \
	dim3 blockDim, gridDim; \
	blockDim.x = (GRIDS_X / TILE_X); \
	blockDim.y = (THREADS_X / TILE_X); \
	gridDim.x  = (GRIDS_X / blockDim.x); \
	gridDim.y  = (GRIDS_X * GRIDS_X * GRIDS_X) / (blockDim.x * blockDim.y * (GRIDS_X / blockDim.x)); \

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