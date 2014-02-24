/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 01, 2014
* <Last Time>     Feb 17, 2014
* <File Name>     CUDABasicKernels.cpp
*/

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "CUDAMacroDef.h"
#include "DataStructures.h"
#include "CUDAInterfaces.h"
#include "CUDAMathLib.h"

using namespace sge;

inline __host__ __device__
double atCell( double const *grid, int const x, int const y, int const z )
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_tailer ) return 0.f;
	if ( y > gst_tailer ) return 0.f;
	if ( z > gst_tailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

inline __host__ __device__
void vertices
( double *c000, double *c001, double *c011, double *c010,
double *c100, double *c101, double *c111, double *c110,
double const *grid, double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	*c000 = atCell ( grid, i, j, k );
	*c001 = atCell ( grid, i, j+1, k );
	*c011 = atCell ( grid, i, j+1, k+1 );
	*c010 = atCell ( grid, i, j, k+1 );
	*c100 = atCell ( grid, i+1, j, k );
	*c101 = atCell ( grid, i+1, j+1, k );
	*c111 = atCell ( grid, i+1, j+1, k+1 );
	*c110 = atCell ( grid, i+1, j, k+1 );
}

inline __host__ __device__
double trilinear( double const *grid, double const x, double const y, double const z )
{
	double v000, v001, v010, v011, v100, v101, v110, v111;
	vertices( &v000, &v001, &v011, &v010,
		&v100, &v101, &v111, &v110,
		grid, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = v000 * ( 1 - dx ) + v001 * dx;
	double c10 = v010 * ( 1 - dx ) + v011 * dx;
	double c01 = v100 * ( 1 - dx ) + v101 * dx;
	double c11 = v110 * ( 1 - dx ) + v111 * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

__global__
void kernelPickData( unsigned char *data, double const *grid )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] = (unsigned char) temp;
};

__global__
void kernelPickData
( unsigned char *data, const double *bufs, int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	double value = bufs[ Index(i, j, k) ];

	/* append data to volume data */
	int temp = sground ( value );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};

/* �ɼ��������ݣ���ת��Ϊvolumetric data */
__host__ void hostPickData( SGUCHAR *data, const double *bufs, SGINT3 *nodeIX )
{
	cudaDeviceDim3D();

	nodeIX->x *= GRIDS_X;
	nodeIX->y *= GRIDS_X;
	nodeIX->z *= GRIDS_X;

	kernelPickData cudaDevice(gridDim, blockDim)
		( data, bufs, nodeIX->x, nodeIX->y, nodeIX->z );
};

__global__ void kernelCopyBuffer( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ void kernelSwapBuffer( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__global__ void kernelZeroBuffer( double *grid )
{
	GetIndex ();
	grid [ Index(i,j,k) ] = 0.f;
};

__host__ void hostZeroBuffer( double *grid )
{
	cudaDeviceDim3D();

	kernelZeroBuffer<<<gridDim, blockDim>>> ( grid );
};

__host__ void hostSwapBuffer( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
};

__global__ void kernelBoundary( double *grid, int const cd )
{
	GetIndex();
	BeginSimArea();
	// ...
	EndSimArea();
};

__global__
void kernelJacobi
( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
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

__host__
void hostJacobi
( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim) (grid_out, grid_in, cd, diffusion, divisor);
		kernelBoundary cudaDevice(gridDim, blockDim) (grid_out, cd);
	}
};

__global__ 
void kernelGridAdvection
( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;
	grid_out [ Index(i,j,k) ] = trilinear ( grid_in, u, v, w );

	EndSimArea();
};

__host__
void hostAdvection
( double *grid_out, double const *grid_in, int const cd,
double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim) ( grid_out, grid_in, u_in, v_in, w_in );
	kernelBoundary cudaDevice(gridDim, blockDim) ( grid_out, cd );

};

__host__ 
void hostDiffusion( double *grid_out, double const *grid_in, int const cd, double const diffusion )
{
//	double rate = diffusion * Grids_X * Grids_X * Grids_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, cd, rate, 1+6*rate );
};

__global__
void kernelGradient
( double *div, double *p, double const *vel_u, double const *vel_v, double const *vel_w )
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

__global__
void kernelSubtract
( double *vel_u, double *vel_v, double *vel_w, double const *p )
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
void hostProject
( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) (div, p, vel_u, vel_v, vel_w);
	kernelBoundary cudaDevice(gridDim, blockDim) (div, DENSITY_FIELD);
	kernelBoundary cudaDevice(gridDim, blockDim) (p, DENSITY_FIELD);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 0, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) (vel_u, vel_v, vel_w, p);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_u, VELOCITY_FIELD_U);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_v, VELOCITY_FIELD_V);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_w, VELOCITY_FIELD_W);
};

/*********************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/

__global__ 	
void kernelAddSource ( double *grid, int const number )
{
	GetIndex();
	BeginSimArea();

	const int half = GRIDS_X / 2;

	switch ( number )
	{
	case DENSITY_FIELD: // density
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = 100.f;
	case VELOCITY_FIELD_V: // velocity v
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = 100.f;

	default: // add external force if need
		break;
	}

	EndSimArea();
};

__host__
void hostAddSource ( double *dens, double *vel_u, double *vel_v, double *vel_w  )
{
	cudaDeviceDim3D();

	if ( dens != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( dens, DENSITY_FIELD );
	if ( vel_v != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( vel_v, VELOCITY_FIELD_V );
};


__host__
void VelocitySolver
	( double *dev_u, double *dev_v, double *dev_w, 
	double *dev_u0, double *dev_v0, double *dev_w0,
	double *dev_div, double *dev_p )
{
	hostAddSource ( NULL, NULL, dev_v, NULL );

	// diffuse the velocity field (per axis):
	hostDiffusion ( dev_u0, dev_u, VELOCITY_FIELD_U, VISOCITY );
	hostDiffusion ( dev_v0, dev_v, VELOCITY_FIELD_V, VISOCITY );
	hostDiffusion ( dev_w0, dev_w, VELOCITY_FIELD_W, VISOCITY );
	hostSwapBuffer ( dev_u0, dev_u );
	hostSwapBuffer ( dev_v0, dev_v );
	hostSwapBuffer ( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject ( dev_u, dev_v, dev_w, dev_div, dev_p );
	
	// advect the velocity field (per axis):
	hostAdvection ( dev_u0, dev_u, VELOCITY_FIELD_U, dev_u, dev_v, dev_w );
	hostAdvection ( dev_v0, dev_v, VELOCITY_FIELD_V, dev_u, dev_v, dev_w );
	hostAdvection ( dev_w0, dev_w, VELOCITY_FIELD_W, dev_u, dev_v, dev_w );
	hostSwapBuffer ( dev_u0, dev_u );
	hostSwapBuffer ( dev_v0, dev_v );
	hostSwapBuffer ( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject ( dev_u, dev_v, dev_w, dev_div, dev_p );
};

__host__
void DensitySolver
	( double *dev_den, double *dev_u, double *dev_v, double *dev_w, double *dev_den0 )
{
	hostAddSource ( dev_den, NULL, NULL, NULL );
	hostDiffusion ( dev_den0, dev_den, DENSITY_FIELD, DIFFUSION );
	hostSwapBuffer ( dev_den0, dev_den );
	hostAdvection ( dev_den, dev_den0, DENSITY_FIELD, dev_u, dev_v, dev_w );
};