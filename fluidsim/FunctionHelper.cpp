/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 05, 2014
* <File Name>     FunctionHelper.cpp
*/

#include "FunctionHelper.h"
#include "CUDAMacroDef.h"
#include "CUDAInterfaces.h"

using namespace sge;


SGVOID FunctionHelper::CheckRuntimeErrors( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};

SGVOID FunctionHelper::DeviceDim2Dx( dim3 *grid_out, dim3 *block_out )
{
	block_out->x = TILE_X;
	block_out->y = TILE_X;
	grid_out->x  = GRIDS_X / TILE_X;
	grid_out->y  = GRIDS_X / TILE_X;
};

SGVOID FunctionHelper::DeviceDim3Dx( dim3 *gridDim, dim3 *blockDim )
{
	blockDim->x = (GRIDS_X / TILE_X);
	blockDim->y = (THREADS_X / TILE_X);
	gridDim->x  = (GRIDS_X / blockDim->x);
	gridDim->y  = (GRIDS_X * GRIDS_X * GRIDS_X) / 
		(blockDim->x * blockDim->y * (GRIDS_X / blockDim->x));
};

SGRUNTIMEMSG FunctionHelper::CreateDoubleBuffers( SGINT size, SGINT nPtrs, ... )
{
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		if ( cudaMalloc( (void**)ptr, sizeof(double) * size ) != cudaSuccess )
		{
			CheckRuntimeErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateIntegerBuffers( SGINT size, SGINT nPtrs, ... )
{
	int **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, int** );
		if ( cudaMalloc( (void**)ptr, sizeof(int) * size ) != cudaSuccess )
		{
			CheckRuntimeErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

void FunctionHelper::CopyBuffers( SGSIMPLENODES *bufs, const SGCUDANODES *nodes, SGFIELDTYPE type )
{
	hostCopyBuffer( bufs, nodes, type );
};

void FunctionHelper::CopyBuffers( SGCUDANODES *nodes, const SGSIMPLENODES *bufs, SGFIELDTYPE type )
{
	hostCopyBuffer( nodes, bufs, type );
};

SGRUNTIMEMSG FunctionHelper::CreateHostNodeBuffers( SGHOSTNODE  **node )
{
	*node = (SGHOSTNODE*)malloc( sizeof(SGHOSTNODE));

	if ( *node not_eq nullptr )
		return SG_RUNTIME_OK;
	
	return SG_MALLOC_SPACE_FAILED;
};

SGRUNTIMEMSG FunctionHelper::CreateCUDANodeBuffers( SGCUDANODES **node )
{
	if ( cudaMalloc( (void**)node, sizeof(SGCUDANODES)) eqt cudaSuccess )
		return SG_RUNTIME_OK;

	return SG_MALLOC_SPACE_FAILED;
};

SGRUNTIMEMSG FunctionHelper::CreateCUDATempBuffers( SGSIMPLENODES **bufs )
{
	if ( cudaMalloc( (void**)bufs, sizeof(SGSIMPLENODES)) eqt cudaSuccess )
		return SG_RUNTIME_OK;

	return SG_MALLOC_SPACE_FAILED;
};