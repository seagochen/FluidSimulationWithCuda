/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 17, 2014
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

SGRUNTIMEMSG FunctionHelper::CreateDoubleBuffers( size_t size, SGINT nPtrs, ... )
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

SGRUNTIMEMSG FunctionHelper::CreateIntegerBuffers( size_t size, SGINT nPtrs, ... )
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

SGRUNTIMEMSG FunctionHelper::CreateVolumetricBuffers( size_t size, SGUCHAR **hostbuf, SGUCHAR **devbuf )
{
	*hostbuf = (SGUCHAR*)malloc( sizeof(SGUCHAR) * size );

	if ( *hostbuf eqt nullptr )
		return SG_MALLOC_SPACE_FAILED;

	if ( cudaMalloc( (void**)devbuf, sizeof(SGUCHAR) * size ) not_eq cudaSuccess )
	{
		CheckRuntimeErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
		return SG_MALLOC_SPACE_FAILED;
	}

	return SG_RUNTIME_OK;
};

SGRUNTIMEMSG FunctionHelper::CreateHostBuffers( size_t size, SGINT nPtrs, ... )
{
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		*ptr = (double*)malloc( sizeof(double) * size );
		
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
	double **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );

		if ( cudaMalloc( (void**)ptr, sizeof(double) * size) not_eq cudaSuccess )
		{
			CheckRuntimeErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
			return SG_MALLOC_SPACE_FAILED;
		}
	}
	va_end( ap );

	return SG_RUNTIME_OK;
};