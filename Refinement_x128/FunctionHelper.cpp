/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 26, 2014
* <File Name>     FunctionHelper.cpp
*/

#include "MacroDefinition.h"
#include "FunctionHelper.h"
#include <iostream>

using namespace sge;

using std::cout;
using std::endl;


SGBOOLEAN FunctionHelper::CreateCompNodesForDevice( std::vector<double*> *vectBuf, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptr;

		if ( CreateDeviceBuffers( size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptr );
	}

	return true;
};

SGBOOLEAN FunctionHelper::CreateCompNodesForHost( std::vector<double*> *vectBuf, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptr;

		if ( CreateHostBuffers( size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptr );
	}

	return true;
};

SGBOOLEAN FunctionHelper::CreateCompNodesForDevice( std::vector<int*> *vectBuf, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		int *ptr;

		if ( CreateDeviceBuffers( size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptr );
	}

	return true;
};

SGBOOLEAN FunctionHelper::CreateCompNodesForHost( std::vector<int*> *vectBuf, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		int *ptr;

		if ( CreateHostBuffers( size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptr );
	}

	return true;
};

SGBOOLEAN FunctionHelper::CreateCompNodesForDevice( std::vector<unsigned char*> *vectBuf, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		unsigned char *ptr;

		if ( CreateDeviceBuffers( size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptr );
	}

	return true;
};

SGBOOLEAN FunctionHelper::CreateCompNodesForHost( std::vector<unsigned char*> *vectBuf, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		unsigned char *ptr;

		if ( CreateHostBuffers( size, 1, &ptr ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptr );
	}

	return true;
};

SGVOID FunctionHelper::DeviceParamDim( dim3 *gridDim, dim3 *blockDim, SGINT thread, SGINT tile )
{
	if ( tile <= thread )
	{
		blockDim->x = tile;
		blockDim->y = gridDim->x = gridDim->y = 1;
	}
	else
	{
		blockDim->x = thread;
		blockDim->y = 1;
		gridDim->x = tile / thread;
		gridDim->y = 1;
	}
};

SGVOID FunctionHelper::DeviceParamDim( dim3 *gridDim, dim3 *blockDim, SGINT thread, SGINT tilex, SGINT tiley, SGINT gridx, SGINT gridy )
{
	if ( gridx * gridy <= thread )
	{
		blockDim->x = gridx;
		blockDim->y = gridy;
		gridDim->x = 1;
		gridDim->y = 1;
	}
	else
	{
		blockDim->x = tilex;
		blockDim->y = tiley;
		gridDim->x = gridx / tilex;
		gridDim->y = gridy / tiley;
	}
};

SGVOID FunctionHelper::DeviceParamDim( dim3 *gridDim, dim3 *blockDim, SGINT thread, SGINT tilex, SGINT tiley, SGINT gridx, SGINT gridy, SGINT gridz )
{
	if ( gridx * gridy <= thread )
	{
		blockDim->x = gridx;
		blockDim->y = gridy;
		gridDim->x = 1;
		gridDim->y = gridz;
	}
	else
	{
		blockDim->x = tilex;
		blockDim->y = tiley;
		gridDim->x  = gridx / tilex;
		gridDim->y  = gridx * gridy * gridz / ( tilex * tiley * gridDim->x );
	}
};

#if 0
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
#endif

SGBOOLEAN FunctionHelper::GetCUDALastError( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	
	if ( __err != cudaSuccess)
	{
		cout << "Error: " << msg << endl
			<< "error message: " << cudaGetErrorString( __err ) << endl
			<< "at file: " << file << endl
			<< "at line: " << line << endl;
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