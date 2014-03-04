/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 03, 2014
* <File Name>     FunctionHelper.cpp
*/

#include "MacroDefinition.h"
#include "FunctionHelper.h"

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