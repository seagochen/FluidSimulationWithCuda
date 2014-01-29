/**
* <Author>      Orlando Chen
* <First>       Dec 12, 2013
* <Last>		Jan 29, 2013
* <File>        Kernels.cu
*/

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "CUDADef.h"
#include "Parameters.h"
#include "DataStructures.h"
#include "FunctionHelper.h"
#include "InterpKernels.h"

using sge::SGGRIDTYPE;
using sge::SGCUDAGRID;
using sge::CUDAHelper;

CUDAHelper m_cudahelper;

__global__ 
void kernelCopyGrids( double *buf, const SGCUDAGRID *grids, const SGGRIDTYPE type )
{
	GetIndex();

	switch (type)
	{
	case SG_DENS_GRID:
		buf[ Index(i,j,k) ] = grids[ Index(i,j,k) ].den;
		break;
	case SG_VELU_GRID:
		buf[ Index(i,j,k) ] = grids[ Index(i,j,k) ].u;
		break;
	case SG_VELV_GRID:
		buf[ Index(i,j,k) ] = grids[ Index(i,j,k) ].v;
		break;
	case SG_VELW_GRID:
		buf[ Index(i,j,k) ] = grids[ Index(i,j,k) ].w;
		break;
	case SG_DIV_GRID:
		buf[ Index(i,j,k) ] = grids[ Index(i,j,k) ].div;
		break;
	case SG_PRES_GRID:
		buf[ Index(i,j,k) ] = grids[ Index(i,j,k) ].p;
		break;
	default:
		break;
	}
};

__global__
void kernelCopyBuffer( SGCUDAGRID *grids, const double *buf, const SGGRIDTYPE type )
{
	GetIndex();

	switch ( type )
	{
	case SG_DENS_GRID:
		grids[ Index(i,j,k) ].den = buf[ Index(i,j,k) ];
		break;
	case SG_VELU_GRID:
		grids[ Index(i,j,k) ].u = buf[ Index(i,j,k) ];
		break;
	case SG_VELV_GRID:
		grids[ Index(i,j,k) ].v = buf[ Index(i,j,k) ];
		break;
	case SG_VELW_GRID:
		grids[ Index(i,j,k) ].w = buf[ Index(i,j,k) ];
		break;
	case SG_DIV_GRID:
		grids[ Index(i,j,k) ].div = buf[ Index(i,j,k) ];
		break;
	case SG_PRES_GRID:
		grids[ Index(i,j,k) ].p = buf[ Index(i,j,k) ];
		break;
	default:
		break;
	}
};

__global__
void kernelAddSource( double *buf, const SGCUDAGRID *grids, const unsigned volume )
{
	GetIndex();

	if ( grids[ Index(i,j,k) ].obstacle eqt SG_BD_SOURCE )
	{
		buf[ Index(i,j,k) ] = volume;
	}
};

__global__
void kernelJacobi( double *grid_out, double const *grid_in, 
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

__global__ 
void kernelGridAdvection( double *buf, double *dStores, const SGCUDAGRID *grids, const SGGRIDTYPE type )
{
	GetIndex();

	double u = i - grids[ Index(i,j,k) ].u * DELTATIME;
	double v = j - grids[ Index(i,j,k) ].v * DELTATIME;
	double w = k - grids[ Index(i,j,k) ].w * DELTATIME;

	buf[ Index(i,j,k) ] = atomicTrilinear( dStores, grids, type, u, v, w );
};

__host__
void hostPreBasicFluidKernel( double **dStores, int **nStores, int nPtrs, ... )
{
	if ( cudaMalloc( (void**)dStores, sizeof(double) * TPBUFFER_X ) != cudaSuccess )
	{
		m_cudahelper.CheckErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
		goto Error;
	}
	if ( cudaMalloc( (void**)nStores, sizeof(int) * TPBUFFER_X ) != cudaSuccess )
	{
		m_cudahelper.CheckErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
		goto Error;
	}

	va_list ap; double **ptr;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, double** );
		if ( cudaMalloc( (void**)ptr, sizeof(double) * CUBESIZE_X ) != cudaSuccess )
		{
			m_cudahelper.CheckErrors( "malloc temporary stores failed!", __FILE__, __LINE__ );
			goto Error;
		}
	}
	va_end( ap );

	goto Success;

Error:
	cudaFree( *dStores );
	cudaFree( *nStores );
	exit( 1 );

Success:
	;
};

__host__
void hostAddSource( double *buf1, double *buf2, SGCUDAGRID *grids )
{
	cudaDeviceDim3D();

	kernelCopyGrids <<<gridDim, blockDim>>> ( buf1, grids, SG_DENS_GRID );
	kernelCopyGrids <<<gridDim, blockDim>>> ( buf2, grids, SG_VELV_GRID );

	kernelAddSource <<<gridDim, blockDim>>> ( buf1, grids, SOURCE_DENSITY );
	kernelAddSource <<<gridDim, blockDim>>> ( buf2, grids, SOURCE_VELOCITY );

	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, buf1, SG_DENS_GRID );		
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, buf2, SG_VELV_GRID );
};

__host__
void hostJacobi( double *buf_old, double *buf_new, SGCUDAGRID *grids,
	const SGGRIDTYPE type, const double diffusion, const double divisor )
{
	cudaDeviceDim3D();

	kernelCopyGrids <<<gridDim, blockDim>>> ( buf_old, grids, type );
	kernelCopyGrids <<<gridDim, blockDim>>> ( buf_new, grids, type );

	for ( int k = 0; k < 20; k++)
		kernelJacobi <<<gridDim, blockDim>>> ( buf_new, buf_old, diffusion, divisor );

	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, buf_new, type );
};

__host__
void hostDiffusion( double *buf_old, double *buf_new, sge::SGCUDAGRID *grids,
	const SGGRIDTYPE type, const double diffusion )
{
	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	hostJacobi( buf_old, buf_new, grids, type, rate, 1 + 6 * rate );
};

__host__
void hostAdvection( double *buf, double *dStores, const SGCUDAGRID *grids, const SGGRIDTYPE type )
{
	cudaDeviceDim3D();

	kernelCopyGrids <<<gridDim, blockDim>>> ( buf, grids, type );
	kernelGridAdvection <<<gridDim, blockDim>>> ( buf, dStores, grids, type );
};

__global__
void kernelGradient ( double *div, double *p, const SGCUDAGRID *grids )
{
	GetIndex();
	BeginSimArea();
	
	const double h = 1.f / GRIDS_X;

	// previous instantaneous magnitude of velocity gradient 
	//		= (sum of velocity gradients per axis)/2N:
	div [ Index(i,j,k) ] = -0.5f * h * (
			grids[ Index(i+1, j, k) ].u - grids[ Index(i-1, j, k) ].u + // gradient of u
			grids[ Index(i, j+1, k) ].v - grids[ Index(i, j-1, k) ].v + // gradient of v
			grids[ Index(i, j, k+1) ].w - grids[ Index(i, j, k-1) ].w   // gradient of w
		);
	// zero out the present velocity gradient
	p [ Index(i,j,k) ] = 0.f;
	
	EndSimArea();
};

__global__
void kernelSubtract ( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__
void hostProject ( double *vel_u, double *vel_v, double *vel_w,
	double *div, double *p, SGCUDAGRID *grids ) 
{
	cudaDeviceDim3D();

	kernelCopyGrids <<<gridDim, blockDim>>> ( vel_u, grids, SG_VELU_GRID );
	kernelCopyGrids <<<gridDim, blockDim>>> ( vel_v, grids, SG_VELV_GRID );
	kernelCopyGrids <<<gridDim, blockDim>>> ( vel_w, grids, SG_VELW_GRID );
	kernelCopyGrids <<<gridDim, blockDim>>> ( div, grids, SG_DIV_GRID );
	kernelCopyGrids <<<gridDim, blockDim>>> ( p, grids, SG_PRES_GRID );

	kernelGradient <<<gridDim, blockDim>>> ( div, p, grids );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	kernelJacobi <<<gridDim, blockDim>>> ( p, div, 1.f, 6.f );

	// now subtract this gradient from our current velocity field
	kernelSubtract ( vel_u, vel_v, vel_w, p );
	
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, vel_u, SG_VELU_GRID );
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, vel_v, SG_VELV_GRID );
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, vel_w, SG_VELW_GRID );
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, div, SG_DIV_GRID );
	kernelCopyBuffer <<<gridDim, blockDim>>> ( grids, p, SG_PRES_GRID );
};