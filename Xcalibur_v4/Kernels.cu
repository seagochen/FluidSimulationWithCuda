/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 24, 2014
* <File Name>     Kernel.cu
*/

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "Kernels.h"
#include "ISO646.h"

/************************************************************************************
** The following functions are basic, which can be used by GPU device and local    **
** host.                                                                           **
** If a function (kernel) is general type, the prefix of that function will be     **
** marked as for example, _rand(...).                                              **
*************************************************************************************/


inline __host__ __device__ int _rand( int seed )
{
	seed = (69069 * seed + 1);
	return seed;
};

inline __host__ __device__ double _random( int seed ) 
{
	return ( _rand( seed ) & 0xffff ) / (double)0x10000;
};

inline __host__ __device__  double _crandom( int seed )
{
	return 2.0 * ( _random( seed ) - 0.5 );
};

inline __host__ __device__ double _invsqrt( double x ) 
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
};

inline __host__ __device__ double _sqrt( double x )
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1/x;
};

inline __host__ __device__ int _round( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __host__ __device__ int _ceil( double x )
{
	int val = _round(x);
	if (x > 0)
	{
		return (val >= (int)x) ? val : (int)x;
	}
	else
	{
		return ((int)x >= val) ? x : val;
	}
};

inline __host__ __device__  int _floor(double x)
{
	int val = _round(x);
	if (x > 0)
	{
		return (val < (int)x) ? val : x;
	}
	else
	{
		return ((int)x < val) ? x : val;
	}
};

inline __host__ __device__ int _fabs(int value)
{
	return (value >= 0) ? value : -value;
};

inline __host__ __device__ double _fabs(double value)
{
	return (value >= 0.f) ? value : -value;
};


/************************************************************************************
** Define some basic logical operator, and some new data type below.               **
** The following kernels are used for getting thread id and transforming the id to **
** array's index.                                                                  **
*************************************************************************************/


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


/************************************************************************************
** To upgrade node size to bullet for fluid simulation computation, and degrade    **
** the size of bullet to retrieve data back.                                       **
** Something else, basic buffer operation kernels, such as zero, and copy the data **
** are also defined below in this block.                                           **
*************************************************************************************/

__global__ void kernelLoadBullet
	( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;
	
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int ixd = ix( i+1, j+1, k+1, dstx, dsty, dstz );
	int ixs = ix( i, j, k, srcx, srcy, srcz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelLoadBullet
	( double *dst, cdouble *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;
	
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int ixd = ix( i+1, j+1, k+1, dstx, dsty, dstz );
	int ixs = ix( i, j, k, srcx, srcy, srcz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelExitBullet
	( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;	
	_thread( &i, &j, &k, dstx, dsty, dstz );

	int ixs = ix( i+1, j+1, k+1, srcx, srcy, srcz );
	int ixd = ix( i, j, k, dstx, dsty, dstz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelExitBullet
	( double *dst, cdouble *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;	
	_thread( &i, &j, &k, dstx, dsty, dstz );

	int ixs = ix( i+1, j+1, k+1, srcx, srcy, srcz );
	int ixd = ix( i, j, k, dstx, dsty, dstz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelZeroBuffers
	( int *bullet, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else bullet[ind] = 0;
};

__global__ void kernelZeroBuffers
	( double *bullet, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else bullet[ind] = 0.f;
};

__global__ void kernelZeroBuffers
	( uchar *bullet, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else bullet[ind] = 0;
};

__global__ void kernelZeroBuffers
	( int *buf, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else buf[ind] = 0;
};

__global__ void kernelZeroBuffers
	( double *buf, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else buf[ind] = 0.f;
};

__global__ void kernelZeroBuffers
	( uchar *buf, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else buf[ind] = 0;
};

__global__ void kernelCopyBuffers
	( int *dst, cint *src, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers
	( double *dst, cint *src, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers
	( uchar *dst, cint *src, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers
	( int *dst, cint *src, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers
	( double *dst, cdouble *src, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers
	( uchar *dst, uchar *src, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};


/************************************************************************************
** Picking the value from a grid with given position, trilinear interpolation and  **
** wether the index of element is in halo, those kernels are defined during this   **
** block.                                                                          **
*************************************************************************************/

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


/************************************************************************************
** Basic kernels for solving Navier-Stokes equation.                               **
*************************************************************************************/

#define IX(i,j,k) ix(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X)
#define thread() _thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);
#define isbound(i,j,k) atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X)


__global__ void kernelJacobi
	( double *out, cdouble *in, cdouble diffusion, cdouble divisor )
{
	int i, j, k;
	thread();

	if ( isbound( i, j, k ) )
	{
		double dix = ( divisor > 0 ) ? divisor : 1.f;

		out[ IX(i,j,k) ] = ( in[ IX(i,j,k) ] + diffusion * (
			out[ IX(i-1,j,k) ] + out[ IX(i+1,j,k) ] +
			out[ IX(i,j-1,k) ] + out[ IX(i,j+1,k) ] +
			out[ IX(i,j,k-1) ] + out[ IX(i,j,k+1) ]
			) ) / dix;
	}
};


__global__ void kernelAdvection
	( double *out, cdouble *in, cdouble delta,
	cdouble *u, cdouble *v, cdouble *w )
{
	int i, j, k;
	thread();

	if ( isbound( i, j, k ) )
	{
		double velu = i - u[ IX(i,j,k) ] * delta;
		double velv = j - v[ IX(i,j,k) ] * delta;
		double velw = k - w[ IX(i,j,k) ] * delta;

		out[ IX(i,j,k) ] = atomicTrilinear( in, velu, velv, velw, 
			GRIDS_X,GRIDS_X,GRIDS_X );
	}
};

__global__ void kernelGradient( double *div, double *prs,
							   cdouble *u, cdouble *v, cdouble *w )
{
	int i, j, k;
	thread();

	if ( isbound( i, j, k ) )
	{
		cdouble hx = 1.f / (double)GRIDS_X;
		cdouble hy = 1.f / (double)GRIDS_X;
		cdouble hz = 1.f / (double)GRIDS_X;

		// previous instantaneous magnitude of velocity gradient 
		//		= (sum of velocity gradients per axis)/2N:
		div[ IX(i,j,k) ] = -0.5f * (
			hx * ( u[ IX(i+1,j,k) ] - u[ IX(i-1,j,k) ] ) +
			hy * ( v[ IX(i,j+1,k) ] - v[ IX(i,j-1,k) ] ) +
			hz * ( w[ IX(i,j,k+1) ] - w[ IX(i,j,k-1) ] ) );

		// zero out the present velocity gradient
		prs[ IX(i,j,k) ] = 0.f;
	}
};

__global__ void kernelSubtract( double *u, double *v, double *w, double *prs )
{
	int i, j, k;
	thread();

	if ( isbound( i, j, k ) )
	{
		u[ IX(i,j,k) ] -= 0.5f * GRIDS_X * ( prs[ IX(i+1,j,k) ] - prs[ IX(i-1,j,k) ] );
		v[ IX(i,j,k) ] -= 0.5f * GRIDS_X * ( prs[ IX(i,j+1,k) ] - prs[ IX(i,j-1,k) ] );
		w[ IX(i,j,k) ] -= 0.5f * GRIDS_X * ( prs[ IX(i,j,k+1) ] - prs[ IX(i,j,k-1) ] );
	}
};

__global__ void kernelCopyGrids( double *src, cdouble *dst )
{
	int i, j, k; 
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	src[IX(i,j,k)] = dst[IX(i,j,k)];
};

__device__ void atomicDensityObs( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	/* 当前格点有障碍物，且密度大于0 */
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE and grids[IX(i,j,k)] > 0.f )
	{
		int cells  = 0;
		double val = 0; 

		if ( obstacle[IX(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[IX(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) cells++;

		if ( cells > 0 ) val = grids[IX(i,j,k)] / cells;
		else val = 0.f;

		if ( obstacle[IX(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i-1,j,k)] += val;
		if ( obstacle[IX(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i+1,j,k)] += val;
		if ( obstacle[IX(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j-1,k)] += val;
		if ( obstacle[IX(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j+1,k)] += val;
		if ( obstacle[IX(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j,k-1)] += val;
		if ( obstacle[IX(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) grids[IX(i,j,k+1)] += val;

		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__device__ void atomicVelocityObs_U( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[IX(i,j,k)] > 0.f )
		{
			if ( obstacle[IX(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i-1,j,k)] = grids[IX(i-1,j,k)] -  grids[IX(i,j,k)];
		}
		else
		{
			if ( obstacle[IX(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i+1,j,k)] = grids[IX(i+1,j,k)] -  grids[IX(i,j,k)];
		}
		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__device__ void atomicVelocityObs_V( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[IX(i,j,k)] > 0.f )
		{
			if ( obstacle[IX(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j-1,k)] = grids[IX(i,j-1,k)] - grids[IX(i,j,k)];
		}
		else
		{
			if ( obstacle[IX(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j+1,k)] = grids[IX(i,j+1,k)] - grids[IX(i,j,k)];
		}
		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__device__ void atomicVelocityObs_W( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	if ( obstacle[IX(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[IX(i,j,k)] > 0.f )
		{
			if ( obstacle[IX(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j,k-1)] = grids[IX(i,j,k-1)] - grids[IX(i,j,k)];
		}
		else
		{
			if ( obstacle[IX(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK )
				grids[IX(i,j,k+1)] = grids[IX(i,j,k+1)] - grids[IX(i,j,k)];
		}
		grids[IX(i,j,k)] = 0.f;
	}
	}
};

__global__ void kernelObstacle( double *grids, cdouble *obstacle, cint field )
{
	switch( field )
	{
	case MACRO_DENSITY:
		atomicDensityObs( grids, obstacle );
		break;

	case MACRO_VELOCITY_U:
		atomicVelocityObs_U( grids, obstacle );
		break;

	case MACRO_VELOCITY_V:
		atomicVelocityObs_V( grids, obstacle );
		break;

	case MACRO_VELOCITY_W:
		atomicVelocityObs_W( grids, obstacle );
		break;

	default:
		break;
	}
};

__global__ void kernelAddSource( double *density, double *vel_u, double *vel_v, double *vel_w )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	if ( atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X) ) {
	
	cint half = GRIDS_X / 2;

	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
	{
		/* add source to grids */
		density[IX(i,j,k)] = DENSITY;

		/* add velocity to grids */
		if ( i < half )
			vel_u[IX(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
		elif( i >= half )
			vel_u[IX(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;

		vel_v[IX(i,j,k)] = VELOCITY;

		if ( k < half )
			vel_w[IX(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
		elif ( k >= half )
			vel_w[IX(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;
	}
	}
};

__global__ void kernelPickData( uchar *c, cdouble *bufs, int ofi, int ofj, int ofk, cint grids )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	ofi = ofi * grids + i;
	ofj = ofj * grids + j;
	ofk = ofk * grids + k;

	/* zero c first */
	c[ ix(ofi,ofj,ofk,VOLUME_X,VOLUME_X,VOLUME_X) ] = 0;

	/* append c to volume c */
	int temp = _round( bufs[ IX(i, j, k) ] );
	if ( temp > 0 and temp < 250 )
		c [ ix(ofi,ofj,ofk,VOLUME_X,VOLUME_X,VOLUME_X) ] = (uchar) temp;
};

__global__ void kernelInterRootGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	double x = ( pi * GRIDS_X + i ) * rate;
	double y = ( pj * GRIDS_X + j ) * rate;
	double z = ( pk * GRIDS_X + k ) * rate;

	dst[IX(i,j,k)] = atomicTrilinear( src, x, y, z, GRIDS_X, GRIDS_X, GRIDS_X );
};

__global__ void kernelInterLeafGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	int x = _round( ( pi * GRIDS_X + i ) * rate );
	int y = _round( ( pj * GRIDS_X + j ) * rate );
	int z = _round( ( pk * GRIDS_X + k ) * rate );

	dst[IX(x,y,z)] = src[IX(i,j,k)];
};

__global__ void kernelClearHalo( double *grids )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	grids[IX(gst_header,j,k)] = 0.f;
	grids[IX(gst_tailer,j,k)] = 0.f;
	grids[IX(i,gst_header,k)] = 0.f;
	grids[IX(i,gst_tailer,k)] = 0.f;
	grids[IX(i,j,gst_header)] = 0.f;
	grids[IX(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelHandleHalo
	( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back )
{
	int i, j, k;
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	center[IX(gst_header,j,k)] = left[IX(gst_tailer,j,k)];
	center[IX(gst_tailer,j,k)] = right[IX(gst_header,j,k)];
	center[IX(i,gst_tailer,k)] = up[IX(i,gst_header,k)];
    center[IX(i,gst_header,k)] = down[IX(i,gst_tailer,k)];
	center[IX(i,j,gst_tailer)] = front[IX(i,j,gst_header)];
	center[IX(i,j,gst_header)] = back[IX(i,j,gst_tailer)];

/*	c[cudaIndex3D(gst_header,gst_header,k,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_header,k,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_header,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_header,gst_tailer,k,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_tailer,k,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_tailer,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,gst_header,k,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_header,k,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_header,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,gst_tailer,k,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_tailer,k,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_tailer,k,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_header,j,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_header,j,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,j,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_header,j,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_header,j,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,j,sim_tailer,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,j,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,j,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,j,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(gst_tailer,j,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,j,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,j,sim_tailer,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_header,gst_header,VOLUME_X)] = ( c[cudaIndex3D(i,sim_header,gst_header,VOLUME_X)] + c[cudaIndex3D(i,gst_header,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_header,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(i,sim_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(i,gst_header,sim_tailer,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_tailer,gst_header,VOLUME_X)] = ( c[cudaIndex3D(i,sim_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(i,gst_tailer,sim_header,VOLUME_X)] ) / 2.f;
	c[cudaIndex3D(i,gst_tailer,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(i,sim_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(i,gst_tailer,sim_tailer,VOLUME_X)] ) / 2.f;

	c[cudaIndex3D(gst_header,gst_header,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_header,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_header,gst_header,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_header,sim_tailer,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_header,gst_tailer,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_tailer,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_header,gst_tailer,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_header,gst_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,sim_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_header,gst_tailer,sim_tailer,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_header,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_header,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_header,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_header,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_header,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_header,sim_tailer,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_tailer,gst_header,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_tailer,gst_header,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_tailer,sim_header,VOLUME_X)] ) / 3.f;
	c[cudaIndex3D(gst_tailer,gst_tailer,gst_tailer,VOLUME_X)] = ( c[cudaIndex3D(sim_tailer,gst_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,sim_tailer,gst_tailer,VOLUME_X)] + c[cudaIndex3D(gst_tailer,gst_tailer,sim_tailer,VOLUME_X)] ) / 3.f;
*/
};

__global__ void kernelSumDensity( double *share, cdouble *src, cint no )
{
	int i, j, k; 
	_thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);

	share[no] += src[IX(i,j,k)];
};

#undef IX(i,j,k)
#undef thread()
#undef isbound(i,j,k)