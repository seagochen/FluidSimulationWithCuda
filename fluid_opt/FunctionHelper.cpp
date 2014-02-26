/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 17, 2014
* <File Name>     FunctionHelper.cpp
*/

#include "FunctionHelper.h"
#include "CUDAMacroDef.h"

using namespace sge;

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

#pragma region obsoleted functions

SGRUNTIMEMSG FunctionHelper::CreateHostCharBuffers( size_t size, SGINT nPtrs, ... )
{
	char **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, char** );
		*ptr = (char*)malloc( size );
		
		if ( *ptr eqt nullptr )
		{
			printf( "malloc space failed!\n" );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateHostIntegerBuffers( size_t size, SGINT nPtrs, ... )
{
	int **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, int** );
		*ptr = (int*)malloc( size );
		
		if ( *ptr eqt nullptr )
		{
			printf( "malloc space failed!\n" );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateHostDoubleBuffers( size_t size, SGINT nPtrs, ... )
{
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		*ptr = (double*)malloc( size );
		
		if ( *ptr eqt nullptr )
		{
			printf( "malloc space failed!\n" );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateDeviceCharBuffers( size_t size, SGINT nPtrs, ... )
{
	char **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, char** );

		if ( cudaMalloc( (void**)ptr, size) not_eq cudaSuccess )
		{
			GetCUDALastError( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateDeviceIntegerBuffers( size_t size, SGINT nPtrs, ... )
{
	int **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, int** );

		if ( cudaMalloc( (void**)ptr, size) not_eq cudaSuccess )
		{
			GetCUDALastError( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateDeviceDoubleBuffers( size_t size, SGINT nPtrs, ... )
{
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );

		if ( cudaMalloc( (void**)ptr, size) not_eq cudaSuccess )
		{
			GetCUDALastError( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};

SGVOID FunctionHelper::FreeHostCharBuffers( SGINT nPtrs, ... )
{
	char **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, char** );
		SAFE_FREE_PTR( *ptr );
	}
	va_end( ap );
};

SGVOID FunctionHelper::FreeHostIntegerBuffers( SGINT nPtrs, ... )
{
	int **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, int** );
		SAFE_FREE_PTR( *ptr );
	}
	va_end( ap );
};

SGVOID FunctionHelper::FreeHostDoubleBuffers( SGINT nPtrs, ... )
{
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		SAFE_FREE_PTR( *ptr );
	}
	va_end( ap );
};

SGVOID FunctionHelper::FreeDeviceCharBuffers( SGINT nPtrs, ... )
{
	char **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, char** );
		cudaFree( *ptr );
	}
	va_end( ap );
};

SGVOID FunctionHelper::FreeDeviceIntegerBuffers( SGINT nPtrs, ... )
{
	int **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, int** );
		cudaFree( *ptr );
	}
	va_end( ap );
};

SGVOID FunctionHelper::FreeDeviceDoubleBuffers( SGINT nPtrs, ... )
{
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		cudaFree( *ptr );
	}
	va_end( ap );
};

#pragma endregion

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