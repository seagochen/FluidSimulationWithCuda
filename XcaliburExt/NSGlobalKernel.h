/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Mar 26, 2014
* <Last Time>     Mar 26, 2014
* <File Name>     NSGlobalKernel.h
*/

#ifndef __ns_global_kernel_h_
#define __ns_global_kernel_h_

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "ISO646.h"

inline __host__ __device__ int _round( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __device__ void _thread( int *i )
{
	*i = blockIdx.x * blockDim.x + threadIdx.x;
};

inline __device__ void _thread( int *i, int *j )
{
	*i = blockIdx.x * blockDim.x + threadIdx.x;
	*j = blockIdx.y * blockDim.y + threadIdx.y;
};

inline __device__ void _thread
	( int *i, int *j, int *k, cint tilex, cint tiley, cint tilez )
{
	_thread( i, j );
	
	*k = *i + *j * tilex;
	*k = *k / ( tilex * tiley );
	*i = *i % tilex;
	*j = *j % tiley;
};

inline __device__ __host__ int ix( cint x, cint tilex)
{
	if ( x >= tilex or x < 0 ) return -1;
	return x;
};

inline __device__ __host__ int ix( cint i, cint j, cint tilex, cint tiley )
{
	if ( i < 0 or j < 0 ) return -1;

	int x; if ( i >= tilex ) x = tilex - 1;
	int y; if ( j >= tiley ) y = tiley - 1;

	x = i; y = j;
	return x + y * tilex;
};

inline __host__ __device__ int ix
	( cint i, cint j, cint k, cint tilex, cint tiley, cint tilez )
{
	if ( i < 0 or i >= tilex ) return -1;
	if ( j < 0 or j >= tiley ) return -1;
	if ( k < 0 or k >= tilez ) return -1;

	return i + j * tilex + k * tilex * tiley;
};

#define IX(i,j,k) ix(i, j, k, BULLET_X, BULLET_Y, BULLET_Z )
#define thread() _thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X); i++; j++; k++;
#define isbound(i,j,k) atomicIXNotHalo(i, j, k, BULLET_X, BULLET_Y, BULLET_Z )

inline __device__ bool atomicIXNotHalo
	( cint i, cint j, cint k, cint tx, cint ty, cint tz )
{
	if ( i eqt 0 or i eqt tx - 1 ) return false;
	if ( j eqt 0 or j eqt ty - 1 ) return false;
	if ( k eqt 0 or k eqt tz - 1 ) return false;

	return true;
};

__device__ double atomicGetValue
	( cdouble *grid, cint x, cint y, cint z, cint tx, cint ty, cint tz )
{
	if ( x < 0 or x >= tx ) return 0.f;
	if ( y < 0 or y >= ty ) return 0.f;
	if ( z < 0 or z >= tz ) return 0.f;

	int ind = ix( x, y, z, tx, ty, tz );
	if ( ind < 0 ) return 0.f;
	else return grid[ind];
};

__device__ double atomicTrilinear
	( cdouble *grid, cdouble x, cdouble y, cdouble z, cint tx, cint ty, cint tz )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	double v000 = atomicGetValue( grid, i, j, k, tx, ty, tz );
	double v001 = atomicGetValue( grid, i, j+1, k, tx, ty, tz );
	double v011 = atomicGetValue( grid, i, j+1, k+1, tx, ty, tz );
	double v010 = atomicGetValue( grid, i, j, k+1, tx, ty, tz );
	double v100 = atomicGetValue( grid, i+1, j, k, tx, ty, tz );
	double v101 = atomicGetValue( grid, i+1, j+1, k, tx, ty, tz );
	double v111 = atomicGetValue( grid, i+1, j+1, k+1, tx, ty, tz );
	double v110 = atomicGetValue( grid, i+1, j, k+1, tx, ty, tz );

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

__global__ void kernelAdvection( double *out, cdouble *in, cdouble delta, cdouble *u, cdouble *v, cdouble *w )
{
	int i, j, k;
	thread();

	if ( isbound( i, j, k ) )
	{
		double velu = i - u[ IX(i,j,k) ] * delta;
		double velv = j - v[ IX(i,j,k) ] * delta;
		double velw = k - w[ IX(i,j,k) ] * delta;

		out[ IX(i,j,k) ] = atomicTrilinear( in, velu, velv, velw, BULLET_X, BULLET_Y, BULLET_Z );
	}
};

#undef thread()
#undef isbound(i,j,k)
#undef IX(i,j,k)

#endif