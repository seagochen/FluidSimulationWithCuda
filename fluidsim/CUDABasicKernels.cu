/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 01, 2014
* <Last Time>     Feb 04, 2014
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
	( const SGDOUBLE *buff, const int x, const int y, const int z )
{
	/* �Գ����ü���ڵ㷶Χ�����������⴦���Դ˱��������� */
	if ( x < gst_header or x > gst_tailer ) return 0.f;
	if ( y < gst_header or y > gst_tailer ) return 0.f;
	if ( z < gst_header or z > gst_tailer ) return 0.f;

	return buff[Index(x,y,z)];
};


/* ����������������(x,y,z)����ָ����Ҫ��ȡ���������ͣ��Դ����øü���ڵ����������� */
__device__ void atomicSetValue
	( SGDOUBLE *buff, const SGDOUBLE value, const int x, const int y, const int z )
{
	/* �Գ����ü���ڵ㷶Χ�����������⴦���Դ˱��������� */
	if ( x < gst_header or x > gst_tailer ) return ;
	if ( y < gst_header or y > gst_tailer ) return ;
	if ( z < gst_header or z > gst_tailer ) return ;

	buff[Index(x,y,z)] = value;
};


/* CopyBuffer������֮һ������������ݿ�������ʱ������ */
__global__ void kernelCopyBuffer
	( SGSIMPLENODES *buffs, const SGCUDANODES *nodes, const SGFIELDTYPE type )
{
	GetIndex();

	switch ( type )
	{
	case SG_DENSITY_FIELD:
		buffs->ptrCenter[Index(i,j,k)] = nodes->ptrCenter[Index(i,j,k)].dens;
		buffs->ptrLeft[Index(i,j,k)] = nodes->ptrLeft[Index(i,j,k)].dens;
		buffs->ptrRight[Index(i,j,k)] = nodes->ptrRight[Index(i,j,k)].dens;
		buffs->ptrUp[Index(i,j,k)] = nodes->ptrUp[Index(i,j,k)].dens;
		buffs->ptrDown[Index(i,j,k)] = nodes->ptrDown[Index(i,j,k)].dens;
		buffs->ptrFront[Index(i,j,k)] = nodes->ptrFront[Index(i,j,k)].dens;
		buffs->ptrBack[Index(i,j,k)] = nodes->ptrBack[Index(i,j,k)].dens;
		break;

	case SG_VELOCITY_U_FIELD:
		buffs->ptrCenter[Index(i,j,k)] = nodes->ptrCenter[Index(i,j,k)].u;
		buffs->ptrLeft[Index(i,j,k)] = nodes->ptrLeft[Index(i,j,k)].u;
		buffs->ptrRight[Index(i,j,k)] = nodes->ptrRight[Index(i,j,k)].u;
		buffs->ptrUp[Index(i,j,k)] = nodes->ptrUp[Index(i,j,k)].u;
		buffs->ptrDown[Index(i,j,k)] = nodes->ptrDown[Index(i,j,k)].u;
		buffs->ptrFront[Index(i,j,k)] = nodes->ptrFront[Index(i,j,k)].u;
		buffs->ptrBack[Index(i,j,k)] = nodes->ptrBack[Index(i,j,k)].u;
		break;

	case SG_VELOCITY_V_FIELD:
		buffs->ptrCenter[Index(i,j,k)] = nodes->ptrCenter[Index(i,j,k)].v;
		buffs->ptrLeft[Index(i,j,k)] = nodes->ptrLeft[Index(i,j,k)].v;
		buffs->ptrRight[Index(i,j,k)] = nodes->ptrRight[Index(i,j,k)].v;
		buffs->ptrUp[Index(i,j,k)] = nodes->ptrUp[Index(i,j,k)].v;
		buffs->ptrDown[Index(i,j,k)] = nodes->ptrDown[Index(i,j,k)].v;
		buffs->ptrFront[Index(i,j,k)] = nodes->ptrFront[Index(i,j,k)].v;
		buffs->ptrBack[Index(i,j,k)] = nodes->ptrBack[Index(i,j,k)].v;
		break;

	case SG_VELOCITY_W_FIELD:
		buffs->ptrCenter[Index(i,j,k)] = nodes->ptrCenter[Index(i,j,k)].w;
		buffs->ptrLeft[Index(i,j,k)] = nodes->ptrLeft[Index(i,j,k)].w;
		buffs->ptrRight[Index(i,j,k)] = nodes->ptrRight[Index(i,j,k)].w;
		buffs->ptrUp[Index(i,j,k)] = nodes->ptrUp[Index(i,j,k)].w;
		buffs->ptrDown[Index(i,j,k)] = nodes->ptrDown[Index(i,j,k)].w;
		buffs->ptrFront[Index(i,j,k)] = nodes->ptrFront[Index(i,j,k)].w;
		buffs->ptrBack[Index(i,j,k)] = nodes->ptrBack[Index(i,j,k)].w;
		break;
	}
};


/* CopyBuffer������֮һ������ʱ���ݿ����������� */
__global__ void kernelCopyBuffer
	( SGCUDANODES *nodes, const SGSIMPLENODES *buffs, const SGFIELDTYPE type )
{
	GetIndex();
	
	switch ( type )
	{
	case SG_DENSITY_FIELD:
		nodes->ptrCenter[Index(i,j,k)].dens = buffs->ptrCenter[Index(i,j,k)];
		nodes->ptrLeft[Index(i,j,k)].dens = buffs->ptrLeft[Index(i,j,k)];
		nodes->ptrRight[Index(i,j,k)].dens = buffs->ptrRight[Index(i,j,k)];
		nodes->ptrUp[Index(i,j,k)].dens = buffs->ptrUp[Index(i,j,k)];
		nodes->ptrDown[Index(i,j,k)].dens = buffs->ptrDown[Index(i,j,k)];
		nodes->ptrFront[Index(i,j,k)].dens = buffs->ptrFront[Index(i,j,k)];
		nodes->ptrBack[Index(i,j,k)].dens = buffs->ptrBack[Index(i,j,k)];
		break;

	case SG_VELOCITY_U_FIELD:
		nodes->ptrCenter[Index(i,j,k)].u = buffs->ptrCenter[Index(i,j,k)];
		nodes->ptrLeft[Index(i,j,k)].u = buffs->ptrLeft[Index(i,j,k)];
		nodes->ptrRight[Index(i,j,k)].u = buffs->ptrRight[Index(i,j,k)];
		nodes->ptrUp[Index(i,j,k)].u = buffs->ptrUp[Index(i,j,k)];
		nodes->ptrDown[Index(i,j,k)].u = buffs->ptrDown[Index(i,j,k)];
		nodes->ptrFront[Index(i,j,k)].u = buffs->ptrFront[Index(i,j,k)];
		nodes->ptrBack[Index(i,j,k)].u = buffs->ptrBack[Index(i,j,k)];
		break;

	case SG_VELOCITY_V_FIELD:
		nodes->ptrCenter[Index(i,j,k)].v = buffs->ptrCenter[Index(i,j,k)];
		nodes->ptrLeft[Index(i,j,k)].v = buffs->ptrLeft[Index(i,j,k)];
		nodes->ptrRight[Index(i,j,k)].v = buffs->ptrRight[Index(i,j,k)];
		nodes->ptrUp[Index(i,j,k)].v = buffs->ptrUp[Index(i,j,k)];
		nodes->ptrDown[Index(i,j,k)].v = buffs->ptrDown[Index(i,j,k)];
		nodes->ptrFront[Index(i,j,k)].v = buffs->ptrFront[Index(i,j,k)];
		nodes->ptrBack[Index(i,j,k)].v = buffs->ptrBack[Index(i,j,k)];
		break;

	case SG_VELOCITY_W_FIELD:
		nodes->ptrCenter[Index(i,j,k)].w = buffs->ptrCenter[Index(i,j,k)];
		nodes->ptrLeft[Index(i,j,k)].w = buffs->ptrLeft[Index(i,j,k)];
		nodes->ptrRight[Index(i,j,k)].w = buffs->ptrRight[Index(i,j,k)];
		nodes->ptrUp[Index(i,j,k)].w = buffs->ptrUp[Index(i,j,k)];
		nodes->ptrDown[Index(i,j,k)].w = buffs->ptrDown[Index(i,j,k)];
		nodes->ptrFront[Index(i,j,k)].w = buffs->ptrFront[Index(i,j,k)];
		nodes->ptrBack[Index(i,j,k)].w = buffs->ptrBack[Index(i,j,k)];
		break;
	}
};


/* ��CopyBuffer��C���Է�װ������������ݿ�������ʱ������ */
__host__ void hostCopyBuffer
	( SGSIMPLENODES *buffs, const SGCUDANODES *nodes, const SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelCopyBuffer <<<gridDim, blockDim>>> ( buffs, nodes, type );
};


/* ��CopyBuffer��C���Է�װ������ʱ���ݿ����������� */
__host__ void hostCopyBuffer
	( SGCUDANODES *nodes, const SGSIMPLENODES *buffs, const SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelCopyBuffer <<<gridDim, blockDim>>> ( nodes, buffs, type );
};


/* ��������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
__global__ void kernelSwapBuffer( SGSIMPLENODES *buf1, SGSIMPLENODES *buf2 )
{
	GetIndex ();

	double center = buf1->ptrCenter[Index(i,j,k)];
	double left = buf1->ptrLeft[Index(i,j,k)];
	double right = buf1->ptrRight[Index(i,j,k)];
	double up = buf1->ptrUp[Index(i,j,k)];
	double down = buf1->ptrDown[Index(i,j,k)];
	double front = buf1->ptrFront[Index(i,j,k)];
	double back = buf1->ptrBack[Index(i,j,k)];

	/* swap now */
	buf1->ptrCenter[Index(i,j,k)] = buf2->ptrCenter[Index(i,j,k)];
	buf1->ptrLeft[Index(i,j,k)] = buf2->ptrLeft[Index(i,j,k)];
	buf1->ptrRight[Index(i,j,k)] = buf2->ptrRight[Index(i,j,k)];
	buf1->ptrUp[Index(i,j,k)] = buf2->ptrUp[Index(i,j,k)];
	buf1->ptrDown[Index(i,j,k)] = buf2->ptrDown[Index(i,j,k)];
	buf1->ptrFront[Index(i,j,k)] = buf2->ptrFront[Index(i,j,k)];
	buf1->ptrBack[Index(i,j,k)] = buf2->ptrBack[Index(i,j,k)];

	/* finally */
	buf2->ptrCenter[Index(i,j,k)] = center;
	buf2->ptrLeft[Index(i,j,k)] = left;
	buf2->ptrRight[Index(i,j,k)] = right;
	buf2->ptrUp[Index(i,j,k)] = up;
	buf2->ptrDown[Index(i,j,k)] = down;
	buf2->ptrFront[Index(i,j,k)] = front;
	buf2->ptrBack[Index(i,j,k)] = back;
};


/* ��SwapBuffer��C���Է�װ����������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
__host__ void hostSwapBuffer( SGSIMPLENODES *buf1, SGSIMPLENODES *buf2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer<<<gridDim, blockDim>>>(buf1, buf2);
};


/* ��GPU buffer������������ */
__global__ void kernelZeroBuffer( SGSIMPLENODES *buf )
{
	GetIndex();
	buf->ptrCenter[Index(i,j,k)] = 0.f;
	buf->ptrLeft[Index(i,j,k)]   = 0.f;
	buf->ptrRight[Index(i,j,k)]  = 0.f;
	buf->ptrUp[Index(i,j,k)]     = 0.f;
	buf->ptrDown[Index(i,j,k)]   = 0.f;
	buf->ptrFront[Index(i,j,k)]  = 0.f;
	buf->ptrBack[Index(i,j,k)]   = 0.f;
};


/* ��ZeroBuffer��C���Է�װ����GPU buffer������������ */
__host__ void hostZeroBuffer( SGSIMPLENODES *buf )
{
	cudaDeviceDim3D();
	kernelZeroBuffer<<<gridDim, blockDim>>>( buf );
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
__device__ SGNODECOORD atomicNodeCoord( const int x, const int y, const int z )
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
__device__ double atomicGetDeviceBuffer( const SGSIMPLENODES *buff, const int x, const int y, const int z )
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
		if ( buff->ptrCenter not_eq NULL )
			value = atomicGetValue( buff->ptrCenter, x, y, z );
		break;
	case SG_LEFT:   // λ����ڵ㣬��xֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( buff->ptrLeft not_eq NULL )
			value = atomicGetValue( buff->ptrLeft, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT: // λ���ҽڵ㣬��xֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( buff->ptrRight not_eq NULL )
			value = atomicGetValue( buff->ptrRight, x - GRIDS_X, y, z );
		break;
	case SG_UP:    // λ���Ͻڵ㣬��yֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( buff->ptrUp not_eq NULL )
			value = atomicGetValue( buff->ptrUp, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:  // λ���½ڵ㣬��yֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( buff->ptrDown not_eq NULL )
			value = atomicGetValue( buff->ptrDown, x, y + GRIDS_X, z );
		break;
	case SG_FRONT: // λ��ǰ�ڵ㣬��zֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( buff->ptrFront not_eq NULL )
			value = atomicGetValue( buff->ptrFront, x, y, z - GRIDS_X );
		break;
	case SG_BACK:  // λ�ں�ڵ㣬��zֵǰ�ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( buff->ptrBack not_eq NULL )
			value = atomicGetValue( buff->ptrBack, x, y, z + GRIDS_X );
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
	( SGSIMPLENODES *buff, const double value, const int x, const int y, const int z )
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
		if ( buff->ptrCenter not_eq NULL )
			atomicSetValue( buff->ptrCenter, value, x, y, z );
		break;
	case SG_LEFT:  // λ����ڵ㣬��xֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( buff->ptrLeft not_eq NULL )
			atomicSetValue( buff->ptrLeft, value, x + GRIDS_X, y, z );
		break;
	case SG_RIGHT: // λ���ҽڵ㣬��xֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( buff->ptrRight not_eq NULL )
			atomicSetValue( buff->ptrRight, value, x - GRIDS_X, y, z );
		break;
	case SG_UP:    // λ���Ͻڵ㣬��yֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( buff->ptrUp not_eq NULL )
			atomicSetValue( buff->ptrUp, value, x, y - GRIDS_X, z );
		break;
	case SG_DOWN:  // λ���½ڵ㣬��yֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( buff->ptrDown not_eq NULL )
			atomicSetValue( buff->ptrDown, value, x, y + GRIDS_X, z );
		break;
	case SG_FRONT: // λ��ǰ�ڵ㣬��zֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( buff->ptrFront not_eq NULL )
			atomicSetValue( buff->ptrFront, value, x, y, z - GRIDS_X );
		break;
	case SG_BACK:  // λ�ں�ڵ㣬��zֵǰ�ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( buff->ptrBack not_eq NULL )
			atomicSetValue( buff->ptrBack, value, x, y, z + GRIDS_X );
		break;
	default:
		break;
	}
};

/* ���������ı߽���Ϣ */
__device__ SGBOUNDARY atomicCheckBounds( SGCUDANODES *nodes, const int x, const int y, const int z )
{
	const int upper = GRIDS_X * 2;
	const int lower = -GRIDS_X; 
	
	/* �Գ�����Χ��������д���Ĭ���������ֵ��0.f */
	if ( x < lower or x >= upper ) return ;
	if ( y < lower or y >= upper ) return ;
	if ( z < lower or z >= upper ) return ;

	/* ����ǰ�����껹��������ʹ�ã������Ҫ�ж���������ľ������ڵ��� */
	SGNODECOORD coord = atomicNodeCoord( x, y, z );

	switch (coord)
	{
	case SG_CENTER: // λ�����ģ���˲���Ҫ��ת��
		if ( nodes->ptrCenter not_eq NULL )
			return nodes->ptrCenter[Index(x,y,z)].obstacle;
	case SG_LEFT:  // λ����ڵ㣬��xֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrLeft not_eq NULL )
			return nodes->ptrLeft[Index(x+GRIDS_X,y,z)].obstacle;
	case SG_RIGHT: // λ���ҽڵ㣬��xֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrRight not_eq NULL )
			return nodes->ptrRight[Index(x-GRIDS_X,y,z)].obstacle;
	case SG_UP:    // λ���Ͻڵ㣬��yֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrUp not_eq NULL )
			return nodes->ptrUp[Index(x,y-GRIDS_X,z)].obstacle;
	case SG_DOWN:  // λ���½ڵ㣬��yֵ���ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrDown not_eq NULL )
			return nodes->ptrDown[Index(x,y+GRIDS_X,z)].obstacle;
	case SG_FRONT: // λ��ǰ�ڵ㣬��zֵ���ƣ�ԭ��Χ[GRIDS_X, GRIDS_X*2-1]��������[0��GRIDS_X-1]
		if ( nodes->ptrFront not_eq NULL )
			return nodes->ptrFront[Index(x,y,z-GRIDS_X)].obstacle;
	case SG_BACK:  // λ�ں�ڵ㣬��zֵǰ�ƣ�ԭ��Χ[-GRIDS_X, -1]��������[0��GRIDS_X-1]
		if ( nodes->ptrBack not_eq NULL )
			return nodes->ptrBack[Index(x,y,z+GRIDS_X)].obstacle;

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
	( double *dStores, const SGSIMPLENODES *buff, double const x, double const y, double const z )
{
	int i = sground( x );
	int j = sground( y );
	int k = sground( z );

	v000 = atomicGetDeviceBuffer( buff, i, j, k );
	v001 = atomicGetDeviceBuffer( buff, i, j+1, k );
	v011 = atomicGetDeviceBuffer( buff, i, j+1, k+1 );
	v010 = atomicGetDeviceBuffer( buff, i, j, k+1 );

	v100 = atomicGetDeviceBuffer( buff, i+1, j, k );
	v101 = atomicGetDeviceBuffer( buff, i+1, j+1, k ); 
	v111 = atomicGetDeviceBuffer( buff, i+1, j+1, k+1 );
	v110 = atomicGetDeviceBuffer( buff, i+1, j, k+1 );
};


/* ���������ݲ�ֵ���ڶ������������ղ�ֵ */
__device__ double atomicTrilinear
	( double *dStores, const SGSIMPLENODES *buff, double const x, double const y, double const z )
{
	atomicPickVertices( dStores, buff, x, y, z );

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
__global__ void kernelAddSource( SGDOUBLE *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
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
__host__ void AddSource( SGDOUBLE *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	cudaDeviceDim3D();

	kernelAddSource <<<gridDim, blockDim>>> ( buffer, grids, type );
};


/* ��density�ı߽紦�����������ֵ�ǰ����ʾ�����ϰ���򽫱�Ӧ�ø�ֵ�ڵ�ǰ����ֵ������
��Χ���Ǳ߽�ĸ���У�������ǰ����densityֵ�޸�Ϊ0.f�����ڲ����ڱ߽�ĸ�㣬�򲻴��� */
__device__ void atomicCheckDensity
	( SGSIMPLENODES *buffer, SGCUDANODES *nodes, const int i, const int j, const int k )
{
	int ix = 0;

	/* ���ڴ��ݵ�����ʱ�ڵ���Ϣ��������Ҫ������Ĵ��� */

	if ( atomicCheckBounds( nodes, i+1, j, k ) not_eq SG_WALL ) ix++;
	if ( atomicCheckBounds( nodes, i-1, j, k ) not_eq SG_WALL ) ix++;
	if ( atomicCheckBounds( nodes, i, j+1, k ) not_eq SG_WALL ) ix++;
	if ( atomicCheckBounds( nodes, i, j-1, k ) not_eq SG_WALL ) ix++;
	if ( atomicCheckBounds( nodes, i, j, k+1 ) not_eq SG_WALL ) ix++;
	if ( atomicCheckBounds( nodes, i, j, k-1 ) not_eq SG_WALL ) ix++;
	
	if ( ix eqt 0 )
	{
		buffer->ptrCenter[Index(i,j,k)] = 0.f;
		return;
	}

	double value = buffer->ptrCenter[Index(i,j,k)] / ix;
	double temp  = 0.f;

	if ( atomicCheckBounds( nodes, i+1, j, k ) not_eq SG_WALL )
	{
		temp = atomicGetDeviceBuffer( buffer, i+1, j, k );
		temp = temp + value;
		atomicSetDeviceBuffer( buffer, temp, i+1, j, k );
	}

	if ( atomicCheckBounds( nodes, i-1, j, k ) not_eq SG_WALL )
	{
		temp = atomicGetDeviceBuffer( buffer, i-1, j, k );
		temp = temp + value;
		atomicSetDeviceBuffer( buffer, temp, i-1, j, k );
	}

	if ( atomicCheckBounds( nodes, i, j+1, k ) not_eq SG_WALL )
	{
		temp = atomicGetDeviceBuffer( buffer, i, j+1, k );
		temp = temp + value;
		atomicSetDeviceBuffer( buffer, temp, i, j+1, k );
	}

	if ( atomicCheckBounds( nodes, i, j-1, k ) not_eq SG_WALL )
	{
		temp = atomicGetDeviceBuffer( buffer, i, j-1, k );
		temp = temp + value;
		atomicSetDeviceBuffer( buffer, temp, i, j-1, k );
	}

	if ( atomicCheckBounds( nodes, i, j, k+1 ) not_eq SG_WALL )
	{
		temp = atomicGetDeviceBuffer( buffer, i, j, k+1 );
		temp = temp + value;
		atomicSetDeviceBuffer( buffer, temp, i, j, k+1 );
	}

	if ( atomicCheckBounds( nodes, i, j, k-1 ) not_eq SG_WALL )
	{
		temp = atomicGetDeviceBuffer( buffer, i, j, k-1 );
		temp = temp + value;
		atomicSetDeviceBuffer( buffer, temp, i, j, k-1 );
	}

	/* �������� */
	buffer->ptrCenter[Index(i,j,k)] = 0.f;
};


/* ���ٶȳ���U��V��W�����ϵķ����Ĵ�����һ�µģ��ǽ���ǰ����ֵ�����ǰ��㣬������ֵ�෴��
����Ҫע�����U��V��W��ʾ�ķ����ǲ�һ���ġ� */
__device__ void atomicCheckVelocity_U
	( SGSIMPLENODES *buffer, SGCUDANODES *nodes, const int i, const int j, const int k )
{
	double temp = buffer->ptrCenter[Index(i,j,k)];

	if ( buffer->ptrCenter[Index(i,j,k)] >= 0.f )
	{
		if ( atomicCheckBounds( nodes, i-1, j, k ) not_eq SG_WALL )
		{
			temp -= atomicGetDeviceBuffer( buffer, i-1, j, k );
			atomicSetDeviceBuffer( buffer, temp, i-1, j, k );
		}
	}
	else
	{
		if ( atomicCheckBounds( nodes, i+1, j, k ) not_eq SG_WALL )
		{
			temp -= atomicGetDeviceBuffer( buffer, i+1, j, k );
			atomicSetDeviceBuffer( buffer, temp, i+1, j, k );
		}
	}

	buffer->ptrCenter[Index(i,j,k)] = 0.f;
};


/* ���ٶȳ���U��V��W�����ϵķ����Ĵ�����һ�µģ��ǽ���ǰ����ֵ�����ǰ��㣬������ֵ�෴��
����Ҫע�����U��V��W��ʾ�ķ����ǲ�һ���ġ� */
__device__ void atomicCheckVelocity_V
	( SGSIMPLENODES *buffer, SGCUDANODES *nodes, const int i, const int j, const int k )
{
	double temp = buffer->ptrCenter[Index(i,j,k)];

	if ( buffer->ptrCenter[Index(i,j,k)] >= 0.f )
	{
		if ( atomicCheckBounds( nodes, i, j-1, k ) not_eq SG_WALL )
		{
			temp -= atomicGetDeviceBuffer( buffer, i, j-1, k );
			atomicSetDeviceBuffer( buffer, temp, i, j-1, k );
		}
	}
	else
	{
		if ( atomicCheckBounds( nodes, i, j+1, k ) not_eq SG_WALL )
		{
			temp -= atomicGetDeviceBuffer( buffer, i, j+1, k );
			atomicSetDeviceBuffer( buffer, temp, i, j+1, k );
		}
	}

	buffer->ptrCenter[Index(i,j,k)] = 0.f;
};


/* ���ٶȳ���U��V��W�����ϵķ����Ĵ�����һ�µģ��ǽ���ǰ����ֵ�����ǰ��㣬������ֵ�෴��
����Ҫע�����U��V��W��ʾ�ķ����ǲ�һ���ġ� */
__device__ void atomicCheckVelocity_W
	( SGSIMPLENODES *buffer, SGCUDANODES *nodes, const int i, const int j, const int k )
{
	double temp = buffer->ptrCenter[Index(i,j,k)];

	if ( buffer->ptrCenter[Index(i,j,k)] >= 0.f )
	{
		if ( atomicCheckBounds( nodes, i, j, k-1 ) not_eq SG_WALL )
		{
			temp -= atomicGetDeviceBuffer( buffer, i, j, k-1 );
			atomicSetDeviceBuffer( buffer, temp, i, j, k-1 );
		}
	}
	else
	{
		if ( atomicCheckBounds( nodes, i, j, k+1 ) not_eq SG_WALL )
		{
			temp -= atomicGetDeviceBuffer( buffer, i, j, k+1 );
			atomicSetDeviceBuffer( buffer, temp, i, j, k+1 );
		}
	}

	buffer->ptrCenter[Index(i,j,k)] = 0.f;
};


/* �߽��⣬��ǰ��Ҫ����������ڵ�������Ƿ����˱߽�Խ�粢��Ҫ����ֵ�����⴦��
�����ڸýڵ�����ݿ��ܻ��������ڽ��ڵ㣬�����Ҫ����ȫ���߸��ڵ�����ݣ����Կ��ܷ���
���������ʵʱ�������Ա�֤ģ�͵�׼ȷ�ԡ� */
__global__ void kernelBoundary( SGSIMPLENODES *buffer, SGCUDANODES *nodes, SGFIELDTYPE type )
{
	GetIndex();

	/* ��������Ҫ����������ڵ��������� */
	if ( atomicCheckBounds( nodes, i, j, k ) eqt SG_WALL )
	{
		switch ( type )
		{
		case SG_DENSITY_FIELD:
			atomicCheckDensity( buffer, nodes, i, j, k );
			break;
		case SG_VELOCITY_U_FIELD:
			atomicCheckVelocity_U( buffer, nodes, i, j, k );
			break;
		case SG_VELOCITY_V_FIELD:
			atomicCheckVelocity_V( buffer, nodes, i, j, k );
			break;
		case SG_VELOCITY_W_FIELD:
			atomicCheckVelocity_W( buffer, nodes, i, j, k );
			break;

		default:
			break;
		}
	}
};


__global__ void kernelJacobi
	( SGSIMPLENODES *buf_out, SGSIMPLENODES *buf_in, double diffusion, double divisor )
{
	GetIndex();

	if ( divisor <= 0.f ) divisor = 1.f;

	double in = atomicGetDeviceBuffer( buf_in, i, j, k );
	double i0 = atomicGetDeviceBuffer( buf_out, i-1, j, k );
	double j0 = atomicGetDeviceBuffer( buf_out, i, j-1, k );
	double k0 = atomicGetDeviceBuffer( buf_out, i, j, k-1 );
	double i1 = atomicGetDeviceBuffer( buf_out, i+1, j, k );
	double j1 = atomicGetDeviceBuffer( buf_out, i, j+1, k );
	double k1 = atomicGetDeviceBuffer( buf_out, i, j, k+1 );

	double fin = ( in + diffusion * ( i0 + i1 + j0 + j1 + k0 + k1 ) ) / divisor;

	atomicSetDeviceBuffer( buf_out, fin, i, j, k );
};


__host__ void hostJacobi
	( SGSIMPLENODES *buf_out, SGSIMPLENODES *buf_in, double diffusion,
	SGCUDANODES *nodes, SGFIELDTYPE type )
{
	double rate = diffusion;

	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
		kernelJacobi <<<gridDim, blockDim>>> ( buf_out, buf_in, rate, 1+6*rate );
	kernelBoundary <<<gridDim, blockDim>>> ( buf_out, nodes, type );
};


__global__ void kernelAdvection( double *stores, SGSIMPLENODES *buff, SGCUDANODES *nodes )
{
	GetIndex();

	double u = i - nodes->ptrCenter[Index(i,j,k)].u * DELTATIME;
	double v = j - nodes->ptrCenter[Index(i,j,k)].v * DELTATIME;
	double w = k - nodes->ptrCenter[Index(i,j,k)].w * DELTATIME;

	double fin = atomicTrilinear( stores, buff, u, v, w );
	atomicSetDeviceBuffer( buff, fin, i, j, k );
};


__host__ void hostAdvection( double *stores, SGSIMPLENODES *buff, SGCUDANODES *nodes, SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelAdvection <<<gridDim, blockDim>>> ( stores, buff, nodes );
	kernelBoundary <<<gridDim, blockDim>>> ( buff, nodes, type );
};


__global__ void kernelGradient
	( SGSIMPLENODES *div, SGSIMPLENODES *p, SGSIMPLENODES *u, SGSIMPLENODES *v, SGSIMPLENODES *w )
{
	GetIndex();
	
	const double h = 1.f / GRIDS_X;

	double fin_div, fin_p;
	double u1 = atomicGetDeviceBuffer( u, i+1, j, k );
	double u0 = atomicGetDeviceBuffer( u, i-1, j, k );
	double v1 = atomicGetDeviceBuffer( v, i, j+1, k );
	double v0 = atomicGetDeviceBuffer( v, i, j-1, k );
	double w1 = atomicGetDeviceBuffer( w, i, j, k+1 );
	double w0 = atomicGetDeviceBuffer( w, i, j, k-1 );

	// previous instantaneous magnitude of velocity gradient 
	//		= (sum of velocity gradients per axis)/2N:
	fin_div = -0.5f * h * ( u1 - u0 + v1 - v0 + w1 - w0 );
	
	// zero out the present velocity gradient
	fin_p = 0.f;

	atomicSetDeviceBuffer( div, fin_div, i, j, k );
	atomicSetDeviceBuffer( p, fin_p, i, j, k );
};


__global__ void kernelSubtract( SGSIMPLENODES *u, SGSIMPLENODES *v, SGSIMPLENODES *w, SGSIMPLENODES *p )
{
	GetIndex();

	double fin_u, fin_v, fin_w;

	double pi1 = atomicGetDeviceBuffer( p, i+1, j, k );
	double pi0 = atomicGetDeviceBuffer( p, i-1, j, k );
	double pj1 = atomicGetDeviceBuffer( p, i, j+1, k );
	double pj0 = atomicGetDeviceBuffer( p, i, j-1, k );
	double pk1 = atomicGetDeviceBuffer( p, i, j, k+1 );
	double pk0 = atomicGetDeviceBuffer( p, i, j, k-1 );

	fin_u = atomicGetDeviceBuffer( u, i, j, k );
	fin_v = atomicGetDeviceBuffer( v, i, j, k );
	fin_w = atomicGetDeviceBuffer( w, i, j, k );

	// gradient calculated by neighbors
	fin_u -= 0.5f * GRIDS_X * ( pi1 - pi0 );
	fin_v -= 0.5f * GRIDS_X * ( pj1 - pj0 );
	fin_w -= 0.5f * GRIDS_X * ( pk1 - pk0 );

	// update grids
	atomicSetDeviceBuffer( u, fin_u, i, j, k );
	atomicSetDeviceBuffer( v, fin_v, i, j, k );
	atomicSetDeviceBuffer( w, fin_w, i, j, k );
};


__host__ void hostProject
	( SGSIMPLENODES *u, SGSIMPLENODES *v, SGSIMPLENODES *w, SGSIMPLENODES *div, SGSIMPLENODES *p,
	SGCUDANODES *nodes )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient <<<gridDim, blockDim>>> ( div, p, u, v, w );
	kernelBoundary <<<gridDim, blockDim>>> ( div, nodes, SG_DENSITY_FIELD );
	kernelBoundary <<<gridDim, blockDim>>> ( p, nodes, SG_DENSITY_FIELD );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi( p, div, 1.f, nodes, SG_DENSITY_FIELD );

	// now subtract this gradient from our current velocity field
	kernelSubtract <<<gridDim, blockDim>>> ( u, v, w, p );

	// check bounds of velocity
	kernelBoundary <<<gridDim, blockDim>>> ( u, nodes, SG_VELOCITY_U_FIELD );
	kernelBoundary <<<gridDim, blockDim>>> ( v, nodes, SG_VELOCITY_V_FIELD );
	kernelBoundary <<<gridDim, blockDim>>> ( w, nodes, SG_VELOCITY_W_FIELD );
};

#pragma endregion


#pragma region velocity and density solvers

__host__ void hostVelocitySolver
	( SGSIMPLENODES *u, SGSIMPLENODES *v, SGSIMPLENODES *w, SGSIMPLENODES *div, SGSIMPLENODES *p,
	SGSIMPLENODES *u0, SGSIMPLENODES *v0, SGSIMPLENODES *w0,
	SGCUDANODES *nodes, double *stores )
{
	/* copy data to temporary buffer */
	hostCopyBuffer( u, nodes, SG_VELOCITY_U_FIELD );
	hostCopyBuffer( v, nodes, SG_VELOCITY_V_FIELD );
	hostCopyBuffer( w, nodes, SG_VELOCITY_W_FIELD );

	/* diffuse the velocity field */
	hostJacobi( u0, u, VISOCITY, nodes, SG_VELOCITY_U_FIELD );
	hostJacobi( v0, v, VISOCITY, nodes, SG_VELOCITY_V_FIELD );
	hostJacobi( w0, w, VISOCITY, nodes, SG_VELOCITY_W_FIELD );
	
	/* velocity field updated */
	hostSwapBuffer( u0, u );
	hostSwapBuffer( v0, v );
	hostSwapBuffer( w0, w );
	
	/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
	hostProject( u, v, w, div, p, nodes );

	/* retrieve data */
	hostCopyBuffer( nodes, u, SG_VELOCITY_U_FIELD );
	hostCopyBuffer( nodes, v, SG_VELOCITY_V_FIELD );
	hostCopyBuffer( nodes, w, SG_VELOCITY_W_FIELD );

	/* advect the velocity field (per axis): */
	hostAdvection( stores, u0, nodes, SG_VELOCITY_U_FIELD );
	hostAdvection( stores, v0, nodes, SG_VELOCITY_V_FIELD );
	hostAdvection( stores, w0, nodes, SG_VELOCITY_W_FIELD );

	/* velocity field updated */
	hostSwapBuffer( u0, u );
	hostSwapBuffer( v0, v );
	hostSwapBuffer( w0, w );
	
	/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
	hostProject( u, v, w, div, p, nodes );

	/* retrieve data */
	hostCopyBuffer( nodes, u, SG_VELOCITY_U_FIELD );
	hostCopyBuffer( nodes, v, SG_VELOCITY_V_FIELD );
	hostCopyBuffer( nodes, w, SG_VELOCITY_W_FIELD );
};

__host__ void DensitySolver
	( SGSIMPLENODES *dens, SGSIMPLENODES *dens0, SGCUDANODES *nodes, double *stores )
{
	/* copy data to temporary buffer */
	hostCopyBuffer( dens, nodes, SG_DENSITY_FIELD );

	/* diffusion */
	hostJacobi( dens0, dens, DIFFUSION, nodes, SG_DENSITY_FIELD );

	/* density updated */
	hostCopyBuffer( nodes, dens0, SG_DENSITY_FIELD );

	/* advect density */
	hostAdvection( stores, dens, nodes, SG_DENSITY_FIELD );

	/* retrive data */
	hostCopyBuffer( nodes, dens, SG_DENSITY_FIELD );
};

#pragma endregion


#pragma region externed functions

__global__ void kernelPickData
	( unsigned char *data, const SGSIMPLENODES *bufs,
	int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	double value = atomicGetDeviceBuffer( bufs, i, j, k );

	/* append data to volume data */
	int temp = sground ( value );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};


/* �ɼ��������ݣ���ת��Ϊvolumetric data */
__host__ void hostPickData 
	( unsigned char *data, const SGSIMPLENODES *bufs,
	int const offi, int const offj, int const offk )
{
	cudaDeviceDim3D();
	kernelPickData cudaDevice(gridDim, blockDim) ( data, bufs, offi, offj, offk );
};

#pragma endregion