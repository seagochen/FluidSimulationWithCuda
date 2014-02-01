/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Feb 01, 2014
* <File>        Kernels.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "DataStructures.h"
#include "CUDAFunctionHelper.h"
#include "CUDADef.h"
#include "CUDADeviceBuffOp.h"

using namespace sge;

CUDAFuncHelper m_helper;

__global__ void kernelAddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	GetIndex();

	if ( grids[Index(i,j,k)].obstacle eqt SG_BD_SOURCE )
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

}

__device__ void atomicBoundary( double *buffer, SGFIELDTYPE type, 
							   const int i, const int j, const int k  )
{
	// TODO
};

__global__ void kernelBoundary( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
{
	GetIndex();

	if ( grids[Index(i,j,k)].obstacle eqt SG_BD_WALL )
	{
		switch ( type )
		{
		case SG_DENSITY_FIELD:
			buffer[Index(i,j,k)] = grids[Index(i,j,k)].dens = 0.f;
			break;
		case SG_VELOCITY_U_FIELD:
			buffer[Index(i,j,k)] = grids[Index(i,j,k)].u = 0.f;
			break;
		case SG_VELOCITY_V_FIELD:
			buffer[Index(i,j,k)] = grids[Index(i,j,k)].v = 0.f;;
			break;
		case SG_VELOCITY_W_FIELD:
			buffer[Index(i,j,k)] = grids[Index(i,j,k)].w = 0.f;
			break;
//		case SG_DIVERGENCE_FIELD:
//			buffer[Index(i,j,k)] = grids[Index(i,j,k)].div = 0.f;
//			break;
//		case SG_PRESSURE_FIELD:
//			buffer[Index(i,j,k)] = grids[Index(i,j,k)].p = 0.f;
//			break;

		default:
			break;
		}
	}

	atomicBoundary( buffer, type, i, j, k );
};

__global__ void kernelJacobi( double *grid_out, double const *grid_in, 
							 double const diffusion, double const divisor )
{
	GetIndex();
	BeginSimArea();

	double div = 0.f;
	if ( divisor <= 0.f ) div = 1.f;
	else div = divisor;

	grid_out [ Index(i,j,k) ] = 
		( grid_in [ Index(i,j,k) ] + diffusion * 
			(
				grid_out [ Index(i-1, j, k) ] + grid_out [ Index(i+1, j, k) ] +
				grid_out [ Index(i, j-1, k) ] + grid_out [ Index(i, j+1, k) ] +
				grid_out [ Index(i, j, k-1) ] + grid_out [ Index(i, j, k+1) ]
			) 
		) / div;

	EndSimArea();
}

__host__ void hostJacobi ( double *grid_out, double const *grid_in, double const diffusion )
{
	double rate = diffusion;

	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	for ( int k=0; k<20; k++)
		kernelJacobi <<<gridDim, blockDim>>> (grid_out, grid_in, rate, 1+6*rate);
};

__global__ void kernelAdvection( double *buffer, double *stores,
								const SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	GetIndex();
	
	double u = i - global->ptrCenter[ Index(i,j,k) ].u * DELTATIME;
	double v = j - global->ptrCenter[ Index(i,j,k) ].v * DELTATIME;
	double w = k - global->ptrCenter[ Index(i,j,k) ].w * DELTATIME;
	buffer[ Index(i,j,k) ] = atomicTrilinear( stores, global, type, u, v, w );
};

__host__ void hostAdvection( double *buffer, double *stores,
								const SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	kernelAdvection <<<gridDim, blockDim>>> ( buffer, stores, global, type );
};

__global__ void kernelGradient( double *div, double *p, 
	double const *vel_u, double const *vel_v, double const *vel_w )
{
	GetIndex();
	BeginSimArea();
	
	const double h = 1.f / GRIDS_X;

	// previous instantaneous magnitude of velocity gradient 
	//		= (sum of velocity gradients per axis)/2N:
	div [ Index(i,j,k) ] = -0.5f * h * (
			vel_u [ Index(i+1, j, k) ] - vel_u [ Index(i-1, j, k) ] + // gradient of u
			vel_v [ Index(i, j+1, k) ] - vel_v [ Index(i, j-1, k) ] + // gradient of v
			vel_w [ Index(i, j, k+1) ] - vel_w [ Index(i, j, k-1) ]   // gradient of w
		);
	// zero out the present velocity gradient
	p [ Index(i,j,k) ] = 0.f;
	
	EndSimArea();
};

__global__ void kernelSubtract( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__ void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	dim3 gridDim, blockDim;
	m_helper.DeviceDim3Dx( &gridDim, &blockDim );

	// the velocity gradient
	kernelGradient <<<gridDim, blockDim>>> (div, p, vel_u, vel_v, vel_w);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 1.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract <<<gridDim, blockDim>>> (vel_u, vel_v, vel_w, p);
};

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


namespace sge
{
	__host__ void Boundary( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	};

	__host__ void VelocitySolver( double *u, double *v, double *w, double *div, double *p,
		double *u0, double *v0, double *w0,
		SGDEVICEBUFF *global, double *stores )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
		
		/* copy data to temporary buffer */
		m_helper.CopyData( u, global, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( v, global, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( w, global, SG_VELOCITY_W_FIELD, SG_CENTER );
		
		/* diffuse the velocity field */
		hostJacobi( u0, u, VISOCITY );
		hostJacobi( v0, v, VISOCITY );
		hostJacobi( w0, w, VISOCITY );
		
		kernelSwapBuffer<<<gridDim, blockDim>>> ( u0, u );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( v0, v );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( w0, w );

		/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
		hostProject( u, v, w, div, p );
		
		/* retrieve data */
		m_helper.CopyData( global, u, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( global, v, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( global, w, SG_VELOCITY_W_FIELD, SG_CENTER );
		
		/* advect the velocity field (per axis): */
		hostAdvection( u0, stores, global, SG_VELOCITY_U_FIELD );
		hostAdvection( v0, stores, global, SG_VELOCITY_V_FIELD );
		hostAdvection( w0, stores, global, SG_VELOCITY_W_FIELD );

		/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
		hostProject ( u0, v0, w0, div, p );

		kernelSwapBuffer<<<gridDim, blockDim>>> ( u0, u );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( v0, v );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( w0, w );

		/* retrieve data */
		m_helper.CopyData( global, u, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( global, v, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( global, w, SG_VELOCITY_W_FIELD, SG_CENTER );
	}

	__host__ void DensitySolver( double *u, double *v, double *w, double *dens, double *dens0,
		SGDEVICEBUFF *global, double *stores )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	
		/* copy data to temporary buffer */
		m_helper.CopyData( u, global, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( v, global, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( w, global, SG_VELOCITY_W_FIELD, SG_CENTER );
		m_helper.CopyData( dens, global, SG_DENSITY_FIELD, SG_CENTER );
	
		/* advection */
		hostJacobi( dens0, dens, DIFFUSION );
		hostAdvection( dens, stores, global, SG_DENSITY_FIELD );

		/* retrive data */
		m_helper.CopyData( global, dens, SG_DENSITY_FIELD, SG_CENTER );
	};

	__host__ void AddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
		kernelAddSource <<<gridDim, blockDim>>> ( buffer, grids, type );
	};

};