/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 12, 2014
* <File Name>     FunctionHelper.cpp
*/

#include "MacroDefinition.h"
#include "FunctionHelper.h"

using namespace sge;

SGVOID FunctionHelper::DeviceDim1D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT gridx )
{
	if ( gridx <= thread )
	{
		blockDim->x = gridx;
		blockDim->y = gridDim->x = gridDim->y = 1;
	}
	else
	{
		blockDim->x = thread;
		blockDim->y = 1;
		gridDim->x = gridx / thread;
		gridDim->y = 1;
	}
};

SGVOID FunctionHelper::DeviceDim2D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT tile, SGINT gridx, SGINT gridy )
{
	if ( gridx * gridy < thread )
	{
		blockDim->x = gridx;
		blockDim->y = gridy;
		gridDim->x = 1;
		gridDim->y = 1;
	}
	else
	{
		blockDim->x = tile;
		blockDim->y = tile;
		gridDim->x = gridx / blockDim->x;
		gridDim->y = gridy / blockDim->y;
	}
};

SGVOID FunctionHelper::DeviceDim3D( dim3 *blockDim, dim3 *gridDim, SGINT thread, SGINT tile, SGINT gridx, SGINT gridy, SGINT gridz )
{
	blockDim->x = tile;
	blockDim->y = tile;
	gridDim->x  = gridx / blockDim->x;
	gridDim->y  = gridx * gridy * gridz / ( blockDim->x * blockDim->y * gridDim->x );
};

SGBOOLEAN FunctionHelper::GetCUDALastError( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if ( __err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString( __err ) );
		printf ( "%s \n", msg );
	
		return true;
	}

	return false;
};

SGRUNTIMEMSG FunctionHelper::CreateHostBuffers( size_t size, SGINT nPtrs, ... )
{
	void **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, void** );
		*ptr = (void*)malloc( size );
		
		if ( *ptr eqt nullptr )
		{
			printf( "malloc space failed!\n" );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateDeviceBuffers( size_t size, SGINT nPtrs, ... )
{
	void **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, void** );

		if ( cudaMalloc( (void**)ptr, size) not_eq cudaSuccess )
		{
			GetCUDALastError( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGVOID FunctionHelper::FreeHostBuffers( SGINT nPtrs, ... )
{
	void **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, void** );
		SAFE_FREE_PTR( *ptr );
	}
	va_end( ap );
};

SGVOID FunctionHelper::FreeDeviceBuffers( SGINT nPtrs, ... )
{
	void **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, void** );
		cudaFree( *ptr );
	}
	va_end( ap );
};