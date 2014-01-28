/**
* <Author>      Orlando Chen
* <First>       Dec 12, 2013
* <Last>		Jan 26, 2013
* <File>        BasicFluidKernels.cu
*/

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <SGE\config\ISO646.h>
#include "Parameters.h"
#include "CUDADef.h"
#include "DataStructures.h"
#include "FunctionHelper.h"
#include "BasicFluidKernels.h"

using namespace sge;

CUDAHelper m_cudahelper;

__host__
void hostPreBasicFluidKernel(double **dStores, int **nStores )
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

	goto Success;

Error:
	cudaFree( *dStores );
	cudaFree( *nStores );
	exit( 1 );

Success:
	;
};

__global__
void kernelAddSource( SGCUDAGRID *grid )
{
	GetIndex();
	int ix = Index(i,j,k);

	if ( grid[ix].obstacle eqt SG_BD_SOURCE )
	{
		grid[ix].v   = SOURCE * DELTATIME;
		grid[ix].den = SOURCE;
	}
};

__host__
void hostAddSource( SGCUDAGRID *grid )
{
	dim3 gridDim, blockDim;
	m_cudahelper.DeviceDim3D( &gridDim, &blockDim );

	kernelAddSource<<<gridDim, blockDim>>>( grid );
};

__device__
void atomicJacobi( double *dStores )
{
	jacobi_out = ( jacobi_in + jacobi_dif *
		( jacobi_x0 + jacobi_x1 + 
		jacobi_y0 + jacobi_y1 +
		jacobi_z0 + jacobi_z1 ) ) / jacobi_div;
};

__global__
void kernelJacobi( SGCUDAGRID *grid, double *dStores, 
	int const type, double const diffusion, double const divisor )
{
	GetIndex();
	BeginSimArea();

	jacobi_dif = diffusion;
	jacobi_div = 0.f;
	if ( divisor <= 0.f ) jacobi_div = 1.f;
	else jacobi_div = divisor;
	
	if ( type eqt SG_SOLVE_DENSITY )
	{
		jacobi_in = grid[ Index(i,j,k) ].den;
		jacobi_x0 = grid[ Index(i-1,j,k) ].den0;
		jacobi_x1 = grid[ Index(i+1,j,k) ].den0;
		jacobi_y0 = grid[ Index(i,j-1,k) ].den0;
		jacobi_y1 = grid[ Index(i,j+1,k) ].den0;
		jacobi_z0 = grid[ Index(i,j,k-1) ].den0;
		jacobi_z1 = grid[ Index(i,j,k+1) ].den0;
		atomicJacobi( dStores );
		grid[ Index(i,j,k) ].den0 = jacobi_out;
	}
	elif ( type eqt SG_SOLVE_VELOCITY )
	{
		/* velocity u */
		jacobi_in = grid[ Index(i,j,k) ].u;
		jacobi_x0 = grid[ Index(i-1,j,k) ].u0;
		jacobi_x1 = grid[ Index(i+1,j,k) ].u0;
		jacobi_y0 = grid[ Index(i,j-1,k) ].u0;
		jacobi_y1 = grid[ Index(i,j+1,k) ].u0;
		jacobi_z0 = grid[ Index(i,j,k-1) ].u0;
		jacobi_z1 = grid[ Index(i,j,k+1) ].u0;
		atomicJacobi( dStores );
		grid[ Index(i,j,k) ].u0 = jacobi_out;

		/* velocity v */
		jacobi_in = grid[ Index(i,j,k) ].v;
		jacobi_x0 = grid[ Index(i-1,j,k) ].v0;
		jacobi_x1 = grid[ Index(i+1,j,k) ].v0;
		jacobi_y0 = grid[ Index(i,j-1,k) ].v0;
		jacobi_y1 = grid[ Index(i,j+1,k) ].v0;
		jacobi_z0 = grid[ Index(i,j,k-1) ].v0;
		jacobi_z1 = grid[ Index(i,j,k+1) ].v0;
		atomicJacobi( dStores );
		grid[ Index(i,j,k) ].v0 = jacobi_out;

		/* velocity w */
		jacobi_in = grid[ Index(i,j,k) ].w;
		jacobi_x0 = grid[ Index(i-1,j,k) ].w0;
		jacobi_x1 = grid[ Index(i+1,j,k) ].w0;
		jacobi_y0 = grid[ Index(i,j-1,k) ].w0;
		jacobi_y1 = grid[ Index(i,j+1,k) ].w0;
		jacobi_z0 = grid[ Index(i,j,k-1) ].w0;
		jacobi_z1 = grid[ Index(i,j,k+1) ].w0;
		atomicJacobi( dStores );
		grid[ Index(i,j,k) ].w0 = jacobi_out;
	}

	EndSimArea();
}

__host__
void hostJacobi ( SGCUDAGRID *grid, double *dStores, 
	int const type, double const diffusion, double const divisor )
{
	dim3 gridDim, blockDim;
	m_cudahelper.DeviceDim3D( &gridDim, &blockDim );

	for ( int k = 0; k < 20; k++)
		kernelJacobi<<<gridDim, blockDim>>>( grid, dStores, type, diffusion, divisor );
};