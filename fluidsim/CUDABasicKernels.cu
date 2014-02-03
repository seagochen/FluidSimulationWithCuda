/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 01, 2014
* <Last Time>     Feb 02, 2014
* <File Name>     CUDABasicKernels.cpp
*/

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "CUDAMacroDef.h"
#include "DataStructures.h"
#include "CUDAInterfaces.h"
#include "CUDAMathLib.h"

using namespace sge;

#pragma region basic cuda functions

/* 给定计算网格坐标(x,y,z)，并指定需要获取的数据类型，以此获取该计算节点的网格的属性 */
__device__ double atomicGetValue
( const SGSTDGRID *buff, const SGFIELDTYPE type, const int x, const int y, const int z )
{
	/* 对超出该计算节点范围的坐标做特殊处理，以此避免程序崩溃 */
	if ( x < gst_header or x > gst_tailer ) return 0.f;
	if ( y < gst_header or y > gst_tailer ) return 0.f;
	if ( z < gst_header or z > gst_tailer ) return 0.f;

	switch (type)
	{
	case SG_DENSITY_FIELD:     // 返回网格的密度值
		return buff[ Index(x,y,z) ].dens;
	case SG_VELOCITY_U_FIELD:  // 返回网格的速度场在U方向上的分量值
		return buff[ Index(x,y,z) ].u;
	case SG_VELOCITY_V_FIELD:  // 返回网格的速度场在V方向上的分量值
		return buff[ Index(x,y,z) ].v;
	case SG_VELOCITY_W_FIELD:  // 返回网格的速度场在W方向上的分量值
		return buff[ Index(x,y,z) ].w;
	}
};


/* 给定计算网格坐标(x,y,z)，并指定需要获取的数据类型，以此设置该计算节点的网格的属性 */
__device__ void atomicSetValue( SGSTDGRID *buff, const double value, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	/* 对超出该计算节点范围的坐标做特殊处理，以此避免程序崩溃 */
	if ( x < gst_header or x > gst_tailer ) return ;
	if ( y < gst_header or y > gst_tailer ) return ;
	if ( z < gst_header or z > gst_tailer ) return ;

	switch (type)
	{
	case SG_DENSITY_FIELD:    // 设置网格的密度值
		buff[ Index(x,y,z) ].dens = value;
		break;
	case SG_VELOCITY_U_FIELD: // 设置网格的速度场在U方向上的分量值
		buff[ Index(x,y,z) ].u = value;
		break;
	case SG_VELOCITY_V_FIELD: // 设置网格的速度场在V方向上的分量值
		buff[ Index(x,y,z) ].v = value;
		break;
	case SG_VELOCITY_W_FIELD: // 设置网格的速度场在W方向上的分量值
		buff[ Index(x,y,z) ].w = value;
		break;
	}
};


/* CopyBuffer的重载之一，将网格的数据拷贝到临时数据中 */
__global__ void kernelCopyBuffer
	( double *buff, const SGSTDGRID *grids, const SGFIELDTYPE type )
{
	GetIndex();

	buff[ Index(i,j,k) ] = atomicGetValue( grids, type, i, j, k );
};


/* CopyBuffer的重载之一，将临时数据拷贝到网格中 */
__global__ void kernelCopyBuffer
	( SGSTDGRID *grids, const double *buff, const SGFIELDTYPE type )
{
	GetIndex();

	double value = buff[ Index(i,j,k) ];
	atomicSetValue( grids, value, type, i, j, k );
};


/* 对CopyBuffer的C语言封装，将网格的数据拷贝到临时数据中 */
__host__ void hostCopyBuffer
	( double *buff, const SGSTDGRID *grids, const SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelCopyBuffer <<<gridDim, blockDim>>> ( buff, grids, type );
};


/* 对CopyBuffer的C语言封装，将临时数据拷贝到网格中 */
__host__ void hostCopyBuffer
	( SGSTDGRID *grids, const double *buff, const SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, buff, type );
};


/* 交换两段GPU buffer的数据，需要注意的是这两段数据的长度应该是一样的，是64^3 */
__global__ void kernelSwapBuffer( double *buf1, double *buf2 )
{
	GetIndex ();

	double temp = buf1 [ Index(i,j,k) ];
	buf1 [ Index(i,j,k) ] = buf2 [ Index(i,j,k) ];
	buf2 [ Index(i,j,k) ] = temp;
};


/* 对SwapBuffer的C语言封装，交换两段GPU buffer的数据，需要注意的是这两段数据的长度应该是一样的，是64^3 */
__host__ void hostSwapBuffer( double *buf1, double *buf2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer<<<gridDim, blockDim>>>(buf1, buf2);
};


/* 对GPU buffer的数据做归零 */
__global__ void kernelZeroBuffer( double *buf )
{
	GetIndex();
	buf[Index(i,j,k)] = 0.f;
};


/* 对ZeroBuffer的C语言封装，对GPU buffer的数据做归零 */
__host__ void hostZeroBuffer( int nPtrs, ... )
{
	cudaDeviceDim3D();
	double *temp;

	va_list ap;
	va_start(ap, nPtrs);
	for ( int i = 0; i < nPtrs; i++ )
	{
		temp = va_arg(ap, double*);
		kernelZeroBuffer<<<gridDim, blockDim>>>( temp );
	}
	va_end(ap);
};

#pragma endregion

/*********************************************************************************************************/

#pragma region GPU buffer I/O


/* 根据输入的坐标(x,y,z)位置，判断出该计算节点与中心节点的相对位置。
不过由于尚不确定的原因，目前只能创建最多4^3的节点，但由于模型只求解中心节点，以及中心节点与周围邻近节点，
上、下、左、右、前、后，共计7个节点之间的关系，所以使用的计算方法也相对简单直接。
另外，对于up-left-front, up-right-front, up-left-back, up-right-back, 
down-left-front, down-right-front, down-left-back, down-right-back 这8个节点没有做计算，并将落入
这些节点的坐标标记为no-define，这些是需要注意的。 */
/* <TODO> 未来的工作之一便是想办法突破4^3个节点的限制，尽可能将空间无限延伸。当然，由于程序是32位的原因，
所以有理由怀疑该限制是由于内存地址限制而造成的。*/
__device__ SGNODECOORD atomicNodeCoord
	( const int x,const int y, const int z )
{
	/* center grids */
	if ( x >= 0 and x < GRIDS_X and   // x in [0, GRIDS_X-1]
		y >= 0 and y < GRIDS_X and    // y in [0, GRIDS_X-1]
		z >= 0 and z < GRIDS_X )      // z in [0, GRIDS_X-1]
		return SG_CENTER;

	/* left grids */
	if ( x >= -GRIDS_X and x < 0 and  // x in [-GRIDS_X, -1]
		y >= 0 and y < GRIDS_X  and   // y in [0, GRIDS_X-1]
		z >= 0 and z < GRIDS_X )      // z in [0, GRIDS_X-1]
		return SG_LEFT;

	/* right grids */
	if ( x >= GRIDS_X and x < GRIDS_X * 2 and   // x in [GRIDS_X, 2*GRIDS_X-1]
		y >= 0 and y < GRIDS_X  and             // y in [0, GRIDS_X-1]
		z >= 0 and z < GRIDS_X )                // z in [0, GRIDS_X-1]
		return SG_RIGHT;

	/* up grids */
	if ( x >= 0 and x < GRIDS_X and             // x in [0, GRIDS_X-1]
		y >= GRIDS_X and y < GRIDS_X * 2 and    // y in [GRIDS_X, 2*GRIDS_X-1]
		z >= 0 and z < GRIDS_X )                // z in [0, GRIDS_X-1]
		return SG_UP;

	/* down grids */
	if ( x >= 0 and x < GRIDS_X and   // x in [0, GRIDS_X-1]
		y >= -GRIDS_X and y < 0 and   // y in [-GRIDS_X, -1]
		z >= 0 and z < GRIDS_X )      // z in [0, GRIDS_X-1]
		return SG_DOWN;

	/* front grids */
	if ( x >= 0 and x < GRIDS_X and         // x in [0, GRIDS_X-1]
		y >= 0 and y < GRIDS_X and          // y in [0, GRIDS_X-1]
		z >= GRIDS_X and z < GRIDS_X * 2 )  // z in [GRIDS_X, 2*GRIDS_X-1]
		return SG_FRONT;

	/* back grids */
	if ( x >= 0 and x < GRIDS_X and   // x in [0, GRIDS_X-1]
		y >= 0 and y < GRIDS_X and    // y in [0, GRIDS_X-1]
		z >= -GRIDS_X and z < 0 )     // z in [-GRIDS_X, -1]
		return SG_BACK;

	return SG_NO_DEFINE;
};


/* 将atomicGetValue封装之后并扩展，使得该函数可以自由的访问上、下、左、右、前、后以及中心这7个节点
的数据，这样对于以后节点之间数据互相交换提供可能。 */
__device__ double atomicGetDeviceBuffer
	( const SGCUDANODES *nodes, const SGFIELDTYPE type, const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2; // 设定坐标上限
	const int lower = -GRIDS_X;    // 设定坐标下限
	
	/* 对超出范围的坐标进行处理，默认情况返回值：0.f */
	if ( x < lower or x >= upper ) return 0.f;
	if ( y < lower or y >= upper ) return 0.f;
	if ( z < lower or z >= upper ) return 0.f;

	/* 但当前的坐标还不能立刻使用，因此需要判断坐标落入的具体计算节点中 */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );
	double value = 0.f;
	
	/* 根据具体情况分别考虑，并对坐标做变化，由绝对坐标转换为相对坐标 */
	switch (coord)
	{
	case SG_CENTER: // 位于中心，因此不需要做转换
		if ( nodes->ptrCenter not_eq NULL )
			value = atomicGetValue( nodes->ptrCenter, type, x, y, z );
		break;
	case SG_LEFT:   // 位于左节点，将x值右移，原范围[-GRIDS_X, -1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrLeft not_eq NULL )
			value = atomicGetValue( nodes->ptrLeft, type, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT: // 位于右节点，将x值左移，原范围[GRIDS_X, GRIDS_X*2-1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrRight not_eq NULL )
			value = atomicGetValue( nodes->ptrRight, type, x - GRIDS_X, y, z );
		break;
	case SG_UP:    // 位于上节点，将y值下移，原范围[GRIDS_X, GRIDS_X*2-1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrUp not_eq NULL )
			value = atomicGetValue( nodes->ptrUp, type, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:  // 位于下节点，将y值上移，原范围[-GRIDS_X, -1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrDown not_eq NULL )
			value = atomicGetValue( nodes->ptrDown, type, x, y + GRIDS_X, z );
		break;
	case SG_FRONT: // 位于前节点，将z值后移，原范围[GRIDS_X, GRIDS_X*2-1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrFront not_eq NULL )
			value = atomicGetValue( nodes->ptrFront, type, x, y, z - GRIDS_X );
		break;
	case SG_BACK:  // 位于后节点，将z值前移，原范围[-GRIDS_X, -1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrBack not_eq NULL )
			value = atomicGetValue( nodes->ptrBack, type, x, y, z + GRIDS_X );
		break;

	default:  // 落入预订之外，做特殊处理
		value = 0.f;
		break;
	}

	return value;
};


/* 将atomicSetValue封装之后并扩展，使得该函数可以自由的访问上、下、左、右、前、后以及中心这7个节点
的数据，这样对于以后节点之间数据互相交换提供可能。 */
__device__ void atomicSetDeviceBuffer
	( SGCUDANODES *nodes, const double value, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* 对超出范围的坐标进行处理，默认情况返回值：0.f */
	if ( x < lower or x >= upper ) return ;
	if ( y < lower or y >= upper ) return ;
	if ( z < lower or z >= upper ) return ;

	/* 但当前的坐标还不能立刻使用，因此需要判断坐标落入的具体计算节点中 */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );

	/* 根据具体情况分别考虑，并对坐标做变化，由绝对坐标转换为相对坐标 */
	switch (coord)
	{
	case SG_CENTER: // 位于中心，因此不需要做转换
		if ( nodes->ptrCenter not_eq NULL )
			atomicSetValue( nodes->ptrCenter, value, type, x, y, z );
		break;
	case SG_LEFT:  // 位于左节点，将x值右移，原范围[-GRIDS_X, -1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrLeft not_eq NULL )
			atomicSetValue( nodes->ptrLeft, value, type, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT: // 位于右节点，将x值左移，原范围[GRIDS_X, GRIDS_X*2-1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrRight not_eq NULL )
			atomicSetValue( nodes->ptrRight, value, type, x - GRIDS_X, y, z );
		break;
	case SG_UP:    // 位于上节点，将y值下移，原范围[GRIDS_X, GRIDS_X*2-1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrUp not_eq NULL )
			atomicSetValue( nodes->ptrUp, value, type, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:  // 位于下节点，将y值上移，原范围[-GRIDS_X, -1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrDown not_eq NULL )
			atomicSetValue( nodes->ptrDown, value, type, x, y + GRIDS_X, z );
		break;
	case SG_FRONT: // 位于前节点，将z值后移，原范围[GRIDS_X, GRIDS_X*2-1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrFront not_eq NULL )
			atomicSetValue( nodes->ptrFront, value, type, x, y, z - GRIDS_X );
		break;
	case SG_BACK:  // 位于后节点，将z值前移，原范围[-GRIDS_X, -1]，修正后[0，GRIDS_X-1]
		if ( nodes->ptrBack not_eq NULL )
			atomicSetValue( nodes->ptrBack, value, type, x, y, z + GRIDS_X );
		break;
	default:
		break;
	}
};

#pragma endregion

/*********************************************************************************************************/

#pragma region trilinear interpolation

#define v000  dStores[ 0 ]
#define v001  dStores[ 1 ]
#define v011  dStores[ 2 ]
#define v010  dStores[ 3 ]
#define v100  dStores[ 4 ]
#define v101  dStores[ 5 ]
#define v111  dStores[ 6 ]
#define v110  dStores[ 7 ]


/* 三线性数据插值，第一步，提取样本点 */
__device__ void atomicPickVertices
	( double *dStores, const SGCUDANODES *nodes, const SGFIELDTYPE type,
	double const x, double const y, double const z )
{
	int i = sground( x );
	int j = sground( y );
	int k = sground( z );

	v000 = atomicGetDeviceBuffer( nodes, type, i, j, k );
	v001 = atomicGetDeviceBuffer( nodes, type, i, j+1, k );
	v011 = atomicGetDeviceBuffer( nodes, type, i, j+1, k+1 );
	v010 = atomicGetDeviceBuffer( nodes, type, i, j, k+1 );

	v100 = atomicGetDeviceBuffer( nodes, type, i+1, j, k );
	v101 = atomicGetDeviceBuffer( nodes, type, i+1, j+1, k ); 
	v111 = atomicGetDeviceBuffer( nodes, type, i+1, j+1, k+1 );
	v110 = atomicGetDeviceBuffer( nodes, type, i+1, j, k+1 );
};


/* 三线性数据插值，第二步，计算最终插值 */
__device__ double atomicTrilinear
	( double *dStores, const SGCUDANODES *nodes, const SGFIELDTYPE type,
	double const x, double const y, double const z )
{
	atomicPickVertices( dStores, nodes, type, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = v000 * ( 1 - dx ) + v001 * dx;
	double c10 = v010 * ( 1 - dx ) + v011 * dx;
	double c01 = v100 * ( 1 - dx ) + v101 * dx;
	double c11 = v110 * ( 1 - dx ) + v111 * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

#undef v000
#undef v001
#undef v011
#undef v010
#undef v100
#undef v101
#undef v111
#undef v110

#pragma endregion

/*********************************************************************************************************/

#pragma region basic functions of fluid simulation

/* 向计算网格中加入数据 */
__global__ void kernelAddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	GetIndex();

	if ( grids[Index(i,j,k)].obstacle eqt SG_SOURCE )
	{
		switch ( type )
		{
		case SG_DENSITY_FIELD:
			buffer[Index(i,j,k)] = SOURCE_DENSITY;
			break;
		case SG_VELOCITY_V_FIELD:
			buffer[Index(i,j,k)] = SOURCE_VELOCITY;
			break;

		default:
			break;
		}
	}
};

/* 向计算网格中加入数据 */
__host__ void AddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	cudaDeviceDim3D();

	kernelAddSource <<<gridDim, blockDim>>> ( buffer, grids, type );
};


/* 对density的边界处理方法，若发现当前格点表示的是障碍物，则将本应该赋值于当前格点的值均分至
周围不是边界的格点中，并将当前格点的density值修改为0.f，对于不处在边界的格点，则不处理。 */
__device__ void atomicCheckDensity
	( double *buffer, SGCUDANODES *nodes, const int i, const int j, const int k )
{
	SGSTDGRID *grids = nodes->ptrCenter;

	int ix = 0;

	/* 由于传递的是临时节点信息，所以需要做特殊的处理 */
	
	if ( grids[Index(i+1,j,k)].obstacle not_eq SG_WALL ) ix++;
	if ( grids[Index(i-1,j,k)].obstacle not_eq SG_WALL ) ix++;
	if ( grids[Index(i,j+1,k)].obstacle not_eq SG_WALL ) ix++;
	if ( grids[Index(i,j-1,k)].obstacle not_eq SG_WALL ) ix++;
	if ( grids[Index(i,j,k+1)].obstacle not_eq SG_WALL ) ix++;
	if ( grids[Index(i,j,k-1)].obstacle not_eq SG_WALL ) ix++;

	if ( ix eqt 0 )
	{
		buffer[Index(i,j,k)] = 0.f;
		return;
	}

	if ( grids[Index(i+1,j,k)].obstacle not_eq SG_WALL )
		buffer[Index(i+1,j,k)] += buffer[Index(i,j,k)] / ix;

	if ( grids[Index(i-1,j,k)].obstacle not_eq SG_WALL )
		buffer[Index(i-1,j,k)] += buffer[Index(i,j,k)] / ix;

	if ( grids[Index(i,j+1,k)].obstacle not_eq SG_WALL )
		buffer[Index(i,j+1,k)] += buffer[Index(i,j,k)] / ix;

	if ( grids[Index(i,j-1,k)].obstacle not_eq SG_WALL )
		buffer[Index(i,j-1,k)] += buffer[Index(i,j,k)] / ix;

	if ( grids[Index(i,j,k+1)].obstacle not_eq SG_WALL )
		buffer[Index(i,j,k+1)] += buffer[Index(i,j,k)] / ix;

	if ( grids[Index(i,j,k-1)].obstacle not_eq SG_WALL )
		buffer[Index(i,j,k-1)] += buffer[Index(i,j,k)] / ix;

	buffer[Index(i,j,k)] = 0.f;
};


/* 对速度场在U、V、W方向上的分量的处理是一致的，是将当前格点的值添加至前格点，并将数值相反。
所需要注意的是U、V、W表示的方向是不一样的。 */
__device__ void atomicVelocity_U
	( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	/* U方向向右为正，因此当检测到当前的格点值大于0时，意味着需要将数值相反，
	并赋值给左边的格点，但需要注意该格点是否处于计算节点的左边界 */
	if ( buffer[Index(i,j,k)] >= 0.f )
	{
		if ( grids[Index(i-1,j,k)].obstacle not_eq SG_WALL )
		{
			buffer[Index(i-1,j,k)] += -buffer[Index(i,j,k)];
		}
	}
	else
	{
		if ( grids[Index(i+1,j,k)].obstacle not_eq SG_WALL )
		{
			buffer[Index(i+1,j,k)] += -buffer[Index(i,j,k)];
		}
	}
};


/* 对速度场在U、V、W方向上的分量的处理是一致的，是将当前格点的值添加至前格点，并将数值相反。
所需要注意的是U、V、W表示的方向是不一样的。 */
__device__ void atomicVelocity_V
	( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	if ( buffer[Index(i,j,k)] >= 0.f )
	{
		if ( grids[Index(i,j-1,k)].obstacle not_eq SG_WALL )
		{
			buffer[Index(i,j-1,k)] += -buffer[Index(i,j,k)];
		}
	}
	else
	{
		if ( grids[Index(i,j+1,k)].obstacle not_eq SG_WALL )
		{
			buffer[Index(i,j+1,k)] += -buffer[Index(i,j,k)];
		}
	}

	buffer[Index(i,j,k)] = 0.f;
};


/* 对速度场在U、V、W方向上的分量的处理是一致的，是将当前格点的值添加至前格点，并将数值相反。
所需要注意的是U、V、W表示的方向是不一样的。 */
__device__ void atomicVelocity_W
	( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	if ( buffer[Index(i,j,k)] >= 0.f )
	{
		if ( grids[Index(i,j,k-1)].obstacle not_eq SG_WALL )
		{
			buffer[Index(i,j,k-1)] += -buffer[Index(i,j,k)];
		}
	}
	else
	{
		if ( grids[Index(i,j,k+1)].obstacle not_eq SG_WALL )
		{
			buffer[Index(i,j,k+1)] += -buffer[Index(i,j,k)];
		}
	}

	buffer[Index(i,j,k)] = 0.f;
};


/* 边界检测，当前需要检测的是中央节点的数据是否发生了边界越界并需要对数值做特殊处理，
但由于该节点的数据可能会外溢至邻近节点，因此需要传递全部七个节点的数据，并对可能发生
溢出的数据实时调整，以保证模型的准确性。 */
__global__ void kernelBoundary
	( double *buffer, SGCUDANODES *nodes, SGFIELDTYPE type )
{
	GetIndex();

	/* 这里所需要检测的是中央节点的数据情况 */
	if ( nodes->ptrCenter[Index(i,j,k)].obstacle eqt SG_WALL )
	{
		switch ( type )
		{
//		case SG_DENSITY_FIELD:
//			atomicCheckDensity( buffer, grids, i, j, k );
//			break;
//		case SG_VELOCITY_U_FIELD:
//			atomicVelocity_U( buffer, grids, i, j, k );
//			break;
//		case SG_VELOCITY_V_FIELD:
//			atomicVelocity_V( buffer, grids, i, j, k );
//			break;
//		case SG_VELOCITY_W_FIELD:
//			atomicVelocity_W( buffer, grids, i, j, k );
//			break;
//		default:
//			break;
		}
	}
};

#pragma endregion