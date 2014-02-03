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

/* ����������������(x,y,z)����ָ����Ҫ��ȡ���������ͣ��Դ˻�ȡ�ü���ڵ����������� */
__device__ double atomicGetValue
( const SGSTDGRID *buff, const SGFIELDTYPE type, const int x, const int y, const int z )
{
	/* �Գ����ü���ڵ㷶Χ�����������⴦���Դ˱��������� */
	if ( x < gst_header or x > gst_tailer ) return 0.f;
	if ( y < gst_header or y > gst_tailer ) return 0.f;
	if ( z < gst_header or z > gst_tailer ) return 0.f;

	switch (type)
	{
	case SG_DENSITY_FIELD:     // ����������ܶ�ֵ
		return buff[ Index(x,y,z) ].dens;
	case SG_VELOCITY_U_FIELD:  // ����������ٶȳ���U�����ϵķ���ֵ
		return buff[ Index(x,y,z) ].u;
	case SG_VELOCITY_V_FIELD:  // ����������ٶȳ���V�����ϵķ���ֵ
		return buff[ Index(x,y,z) ].v;
	case SG_VELOCITY_W_FIELD:  // ����������ٶȳ���W�����ϵķ���ֵ
		return buff[ Index(x,y,z) ].w;
	}
};


/* ����������������(x,y,z)����ָ����Ҫ��ȡ���������ͣ��Դ����øü���ڵ����������� */
__device__ void atomicSetValue( SGSTDGRID *buff, const double value, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	/* �Գ����ü���ڵ㷶Χ�����������⴦���Դ˱��������� */
	if ( x < gst_header or x > gst_tailer ) return ;
	if ( y < gst_header or y > gst_tailer ) return ;
	if ( z < gst_header or z > gst_tailer ) return ;

	switch (type)
	{
	case SG_DENSITY_FIELD:    // ����������ܶ�ֵ
		buff[ Index(x,y,z) ].dens = value;
		break;
	case SG_VELOCITY_U_FIELD: // ����������ٶȳ���U�����ϵķ���ֵ
		buff[ Index(x,y,z) ].u = value;
		break;
	case SG_VELOCITY_V_FIELD: // ����������ٶȳ���V�����ϵķ���ֵ
		buff[ Index(x,y,z) ].v = value;
		break;
	case SG_VELOCITY_W_FIELD: // ����������ٶȳ���W�����ϵķ���ֵ
		buff[ Index(x,y,z) ].w = value;
		break;
	}
};


/* CopyBuffer������֮һ������������ݿ�������ʱ������ */
__global__ void kernelCopyBuffer
	( double *buff, const SGSTDGRID *grids, const SGFIELDTYPE type )
{
	GetIndex();

	buff[ Index(i,j,k) ] = atomicGetValue( grids, type, i, j, k );
};


/* CopyBuffer������֮һ������ʱ���ݿ����������� */
__global__ void kernelCopyBuffer
	( SGSTDGRID *grids, const double *buff, const SGFIELDTYPE type )
{
	GetIndex();

	double value = buff[ Index(i,j,k) ];
	atomicSetValue( grids, value, type, i, j, k );
};


/* ��CopyBuffer��C���Է�װ������������ݿ�������ʱ������ */
__host__ void hostCopyBuffer
	( double *buff, const SGSTDGRID *grids, const SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelCopyBuffer <<<gridDim, blockDim>>> ( buff, grids, type );
};


/* ��CopyBuffer��C���Է�װ������ʱ���ݿ����������� */
__host__ void hostCopyBuffer
	( SGSTDGRID *grids, const double *buff, const SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, buff, type );
};


/* ��������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
__global__ void kernelSwapBuffer( double *buf1, double *buf2 )
{
	GetIndex ();

	double temp = buf1 [ Index(i,j,k) ];
	buf1 [ Index(i,j,k) ] = buf2 [ Index(i,j,k) ];
	buf2 [ Index(i,j,k) ] = temp;
};


/* ��SwapBuffer��C���Է�װ����������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
__host__ void hostSwapBuffer( double *buf1, double *buf2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer<<<gridDim, blockDim>>>(buf1, buf2);
};


/* ��GPU buffer������������ */
__global__ void kernelZeroBuffer( double *buf )
{
	GetIndex();
	buf[Index(i,j,k)] = 0.f;
};


/* ��ZeroBuffer��C���Է�װ����GPU buffer������������ */
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


/* �������������(x,y,z)λ�ã��жϳ��ü���ڵ������Ľڵ�����λ�á�
���������в�ȷ����ԭ��Ŀǰֻ�ܴ������4^3�Ľڵ㣬������ģ��ֻ������Ľڵ㣬�Լ����Ľڵ�����Χ�ڽ��ڵ㣬
�ϡ��¡����ҡ�ǰ���󣬹���7���ڵ�֮��Ĺ�ϵ������ʹ�õļ��㷽��Ҳ��Լ�ֱ�ӡ�
���⣬����up-left-front, up-right-front, up-left-back, up-right-back, 
down-left-front, down-right-front, down-left-back, down-right-back ��8���ڵ�û�������㣬��������
��Щ�ڵ��������Ϊno-define����Щ����Ҫע��ġ� */
/* <TODO> δ���Ĺ���֮һ������취ͻ��4^3���ڵ�����ƣ������ܽ��ռ��������졣��Ȼ�����ڳ�����32λ��ԭ��
���������ɻ��ɸ������������ڴ��ַ���ƶ���ɵġ�*/
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


/* ��atomicGetValue��װ֮����չ��ʹ�øú����������ɵķ����ϡ��¡����ҡ�ǰ�����Լ�������7���ڵ�
�����ݣ����������Ժ�ڵ�֮�����ݻ��ཻ���ṩ���ܡ� */
__device__ double atomicGetDeviceBuffer
	( const SGCUDANODES *nodes, const SGFIELDTYPE type, const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2; // �趨��������
	const int lower = -GRIDS_X;    // �趨��������
	
	/* �Գ�����Χ��������д���Ĭ���������ֵ��0.f */
	if ( x < lower or x >= upper ) return 0.f;
	if ( y < lower or y >= upper ) return 0.f;
	if ( z < lower or z >= upper ) return 0.f;

	/* ����ǰ�����껹��������ʹ�ã������Ҫ�ж���������ľ������ڵ��� */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );
	double value = 0.f;
	
	/* ���ݾ�������ֱ��ǣ������������仯���ɾ�������ת��Ϊ������� */
	switch (coord)
	{
	case SG_CENTER: // λ�����ģ���˲���Ҫ��ת��
		if ( nodes->ptrCenter not_eq NULL )
			value = atomicGetValue( nodes->ptrCenter, type, x, y, z );
		break;
	case SG_LEFT:   // λ����ڵ㣬��xֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrLeft not_eq NULL )
			value = atomicGetValue( nodes->ptrLeft, type, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT: // λ���ҽڵ㣬��xֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrRight not_eq NULL )
			value = atomicGetValue( nodes->ptrRight, type, x - GRIDS_X, y, z );
		break;
	case SG_UP:    // λ���Ͻڵ㣬��yֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrUp not_eq NULL )
			value = atomicGetValue( nodes->ptrUp, type, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:  // λ���½ڵ㣬��yֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrDown not_eq NULL )
			value = atomicGetValue( nodes->ptrDown, type, x, y + GRIDS_X, z );
		break;
	case SG_FRONT: // λ��ǰ�ڵ㣬��zֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrFront not_eq NULL )
			value = atomicGetValue( nodes->ptrFront, type, x, y, z - GRIDS_X );
		break;
	case SG_BACK:  // λ�ں�ڵ㣬��zֵǰ�ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrBack not_eq NULL )
			value = atomicGetValue( nodes->ptrBack, type, x, y, z + GRIDS_X );
		break;

	default:  // ����Ԥ��֮�⣬�����⴦��
		value = 0.f;
		break;
	}

	return value;
};


/* ��atomicSetValue��װ֮����չ��ʹ�øú����������ɵķ����ϡ��¡����ҡ�ǰ�����Լ�������7���ڵ�
�����ݣ����������Ժ�ڵ�֮�����ݻ��ཻ���ṩ���ܡ� */
__device__ void atomicSetDeviceBuffer
	( SGCUDANODES *nodes, const double value, const SGFIELDTYPE type,
	const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* �Գ�����Χ��������д���Ĭ���������ֵ��0.f */
	if ( x < lower or x >= upper ) return ;
	if ( y < lower or y >= upper ) return ;
	if ( z < lower or z >= upper ) return ;

	/* ����ǰ�����껹��������ʹ�ã������Ҫ�ж���������ľ������ڵ��� */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );

	/* ���ݾ�������ֱ��ǣ������������仯���ɾ�������ת��Ϊ������� */
	switch (coord)
	{
	case SG_CENTER: // λ�����ģ���˲���Ҫ��ת��
		if ( nodes->ptrCenter not_eq NULL )
			atomicSetValue( nodes->ptrCenter, value, type, x, y, z );
		break;
	case SG_LEFT:  // λ����ڵ㣬��xֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrLeft not_eq NULL )
			atomicSetValue( nodes->ptrLeft, value, type, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT: // λ���ҽڵ㣬��xֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrRight not_eq NULL )
			atomicSetValue( nodes->ptrRight, value, type, x - GRIDS_X, y, z );
		break;
	case SG_UP:    // λ���Ͻڵ㣬��yֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrUp not_eq NULL )
			atomicSetValue( nodes->ptrUp, value, type, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:  // λ���½ڵ㣬��yֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrDown not_eq NULL )
			atomicSetValue( nodes->ptrDown, value, type, x, y + GRIDS_X, z );
		break;
	case SG_FRONT: // λ��ǰ�ڵ㣬��zֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrFront not_eq NULL )
			atomicSetValue( nodes->ptrFront, value, type, x, y, z - GRIDS_X );
		break;
	case SG_BACK:  // λ�ں�ڵ㣬��zֵǰ�ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
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


/* ���������ݲ�ֵ����һ������ȡ������ */
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


/* ���������ݲ�ֵ���ڶ������������ղ�ֵ */
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

/* ����������м������� */
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

/* ����������м������� */
__host__ void AddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	cudaDeviceDim3D();

	kernelAddSource <<<gridDim, blockDim>>> ( buffer, grids, type );
};


/* ��density�ı߽紦�����������ֵ�ǰ����ʾ�����ϰ���򽫱�Ӧ�ø�ֵ�ڵ�ǰ����ֵ������
��Χ���Ǳ߽�ĸ���У�������ǰ����densityֵ�޸�Ϊ0.f�����ڲ����ڱ߽�ĸ�㣬�򲻴��� */
__device__ void atomicCheckDensity
	( double *buffer, SGCUDANODES *nodes, const int i, const int j, const int k )
{
	SGSTDGRID *grids = nodes->ptrCenter;

	int ix = 0;

	/* ���ڴ��ݵ�����ʱ�ڵ���Ϣ��������Ҫ������Ĵ��� */
	
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


/* ���ٶȳ���U��V��W�����ϵķ����Ĵ�����һ�µģ��ǽ���ǰ����ֵ�����ǰ��㣬������ֵ�෴��
����Ҫע�����U��V��W��ʾ�ķ����ǲ�һ���ġ� */
__device__ void atomicVelocity_U
	( double *buffer, SGSTDGRID *grids, const int i, const int j, const int k )
{
	/* U��������Ϊ������˵���⵽��ǰ�ĸ��ֵ����0ʱ����ζ����Ҫ����ֵ�෴��
	����ֵ����ߵĸ�㣬����Ҫע��ø���Ƿ��ڼ���ڵ����߽� */
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


/* ���ٶȳ���U��V��W�����ϵķ����Ĵ�����һ�µģ��ǽ���ǰ����ֵ�����ǰ��㣬������ֵ�෴��
����Ҫע�����U��V��W��ʾ�ķ����ǲ�һ���ġ� */
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


/* ���ٶȳ���U��V��W�����ϵķ����Ĵ�����һ�µģ��ǽ���ǰ����ֵ�����ǰ��㣬������ֵ�෴��
����Ҫע�����U��V��W��ʾ�ķ����ǲ�һ���ġ� */
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


/* �߽��⣬��ǰ��Ҫ����������ڵ�������Ƿ����˱߽�Խ�粢��Ҫ����ֵ�����⴦��
�����ڸýڵ�����ݿ��ܻ��������ڽ��ڵ㣬�����Ҫ����ȫ���߸��ڵ�����ݣ����Կ��ܷ���
���������ʵʱ�������Ա�֤ģ�͵�׼ȷ�ԡ� */
__global__ void kernelBoundary
	( double *buffer, SGCUDANODES *nodes, SGFIELDTYPE type )
{
	GetIndex();

	/* ��������Ҫ����������ڵ��������� */
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