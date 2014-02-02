/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 01, 2014
* <File Name>     CUDAFunctionHelper.cpp
*/

namespace sge
{

SGVOID CUDAFuncHelper::CheckRuntimeErrors( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};

SGVOID CUDAFuncHelper::DeviceDim2Dx( dim3 *grid_out, dim3 *block_out )
{
	block_out->x = TILE_X;
	block_out->y = TILE_X;
	grid_out->x  = GRIDS_X / TILE_X;
	grid_out->y  = GRIDS_X / TILE_X;
};

SGVOID CUDAFuncHelper::DeviceDim3Dx( dim3 *gridDim, dim3 *blockDim )
{
	blockDim->x = (GRIDS_X / TILE_X);
	blockDim->y = (THREADS_X / TILE_X);
	gridDim->x  = (GRIDS_X / blockDim->x);
	gridDim->y  = (GRIDS_X * GRIDS_X * GRIDS_X) / 
		(blockDim->x * blockDim->y * (GRIDS_X / blockDim->x));
};

SGRUNTIMEMSG CUDAFuncHelper::CreateDoubleBuffers( SGINT size, SGINT nPtrs, ... )
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

SGRUNTIMEMSG CUDAFuncHelper::CreateIntegerBuffers( SGINT size, SGINT nPtrs, ... )
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

void CUDAFuncHelper::CopyData
	( SGDOUBLE *buffer, const SGDEVICEBUFF *devbuffs, SGFIELDTYPE type, SGNODECOORD coord )
{
	dim3 grid, block;
	DeviceDim3Dx( &grid, &block );
	switch ( coord )
	{
	case SG_CENTER:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrCenter, type );
		break;
	case SG_LEFT:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrLeft, type );
		break;
	case SG_RIGHT:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrRight, type );
		break;
	case SG_UP:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrUp, type );
		break;
	case SG_DOWN:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrDown, type );
		break;
	case SG_FRONT:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrFront, type );
		break;
	case SG_BACK:
		kernelCopyBuffer <<<grid, block>>> ( buffer, devbuffs->ptrBack, type );
		break;

	default:
		break;
	}
};

void CUDAFuncHelper::CopyData
	( SGDEVICEBUFF *devbuffs, const SGDOUBLE *buffer, SGFIELDTYPE type, SGNODECOORD coord )
{
	dim3 grid, block;
	DeviceDim3Dx( &grid, &block );
	switch ( coord )
	{
	case SG_CENTER:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrCenter, buffer, type );
		break;
	case SG_LEFT:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrLeft, buffer, type );
		break;
	case SG_RIGHT:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrRight, buffer, type );
		break;
	case SG_UP:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrUp, buffer, type );
		break;
	case SG_DOWN:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrDown, buffer, type );
		break;
	case SG_FRONT:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrFront, buffer, type );
		break;
	case SG_BACK:
		kernelCopyBuffer <<<grid, block>>> ( devbuffs->ptrBack, buffer, type );
		break;

	default:
		break;
	}
};

};