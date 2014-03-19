/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Mar 19, 2014
* <File Name>     Kernel.cu
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

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




#define and    &&
#define and_eq &=
#define bitand &
#define bitor  |
#define compl  ~
#define not    !
#define not_eq !=
#define or     ||
#define or_eq  |=
#define xor    ^
#define xor_eq ^=
#define eqt    ==
#define elif  else if

typedef int const cint;
typedef double const cdouble;
typedef unsigned char uchar;

inline __device__ void _thread( int *i )
{
	*i = blockIdx.x * blockDim.x + threadIdx.x;
};

inline __device__ void _thread( int *i, int *j )
{
	*i = blockIdx.x * blockDim.x + threadIdx.x;
	*j = blockIdx.y * blockDim.y + threadIdx.y;
};

inline __device__ void _thread( int *i, int *j, int *k, cint tilex, cint tiley, cint tilez )
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

inline __host__ __device__ int ix( cint i, cint j, cint k, cint tilex, cint tiley, cint tilez )
{
	if ( i < 0 or i >= tilex ) return -1;
	if ( j < 0 or j >= tiley ) return -1;
	if ( k < 0 or k >= tilez ) return -1;

	return i + j * tilex + k * tilex * tiley;
};

__global__ void kernelLoadBullet( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;
	
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int ixd = ix( i+1, j+1, k+1, dstx, dsty, dstz );
	int ixs = ix( i, j, k, srcx, srcy, srcz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelLoadBullet( double *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;
	
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int ixd = ix( i+1, j+1, k+1, dstx, dsty, dstz );
	int ixs = ix( i, j, k, srcx, srcy, srcz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelExitBullet( int *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;	
	_thread( &i, &j, &k, dstx, dsty, dstz );

	int ixs = ix( i+1, j+1, k+1, srcx, srcy, srcz );
	int ixd = ix( i, j, k, dstx, dsty, dstz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelExitBullet( double *dst, cint *src, cint dstx, cint dsty, cint dstz, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;	
	_thread( &i, &j, &k, dstx, dsty, dstz );

	int ixs = ix( i+1, j+1, k+1, srcx, srcy, srcz );
	int ixd = ix( i, j, k, dstx, dsty, dstz );

	if ( ixd < 0 or ixs < 0 ) return;
	else dst[ixd] = src[ixs];
};

__global__ void kernelZeroBuffers( int *bullet, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else bullet[ind] = 0;
};

__global__ void kernelZeroBuffers( double *bullet, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else bullet[ind] = 0.f;
};

__global__ void kernelZeroBuffers( uchar *bullet, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else bullet[ind] = 0;
};

__global__ void kernelZeroBuffers( int *buf, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else buf[ind] = 0;
};

__global__ void kernelZeroBuffers( double *buf, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else buf[ind] = 0.f;
};

__global__ void kernelZeroBuffers( uchar *buf, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else buf[ind] = 0;
};

__global__ void kernelCopyBuffers( int *dst, cint *src, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers( double *dst, cint *src, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers( uchar *dst, cint *src, cint tiles )
{
	int x;
	_thread( &x );
	int ind = ix( x, tiles );
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers( int *dst, cint *src, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers( double *dst, cdouble *src, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};

__global__ void kernelCopyBuffers( uchar *dst, uchar *src, cint tilex, cint tiley, cint tilez )
{
	int i, j, k;
	_thread( &i, &j, &k, tilex, tiley, tilez );
	int ind = ix( i, j, k, tilex, tiley, tilez );
	
	if ( ind < 0 ) return;
	else dst[ind] = src[ind];
};


#include "MacroDefinition.h"

#define Index(i,j,k) ix(i,j,k,GRIDS_X,GRIDS_Y,GRIDS_Z)

__device__ double atomicGetValue( cdouble *grid, cint x, cint y, cint z, cint tx, cint ty, cint tz )
{
	if ( x < 0 or x >= tx ) return 0.f;
	if ( y < 0 or y >= ty ) return 0.f;
	if ( z < 0 or z >= tz ) return 0.f;

	int ind = ix( x, y, z, tx, ty, tz );
	if ( ind < 0 ) return 0.f;
	else return grid[ind];
};

__device__ double atomicTrilinear( cdouble *grid, cdouble x, cdouble y, cdouble z, cint tx, cint ty, cint tz )
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

inline __device__ bool atomicIsNotHalo( cint i, cint j, cint k, cint tx, cint ty, cint tz )
{
	if ( i eqt 0 or i eqt tx - 1 ) return false;
	if ( j eqt 0 or j eqt ty - 1 ) return false;
	if ( k eqt 0 or k eqt tz - 1 ) return false;

	return true;
};

__device__ void atomicDensityObs( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		/* 当前格点有障碍物，且密度大于0 */
		if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE and grids[Index(i,j,k)] > 0.f )
		{
			int cells  = 0;
			double val = 0; 

			if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
			if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
			if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
			if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
			if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) cells++;
			if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) cells++;

			if ( cells > 0 ) val = grids[Index(i,j,k)] / cells;
			else val = 0.f;

			if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i-1,j,k)] += val;
			if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i+1,j,k)] += val;
			if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j-1,k)] += val;
			if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j+1,k)] += val;
			if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j,k-1)] += val;
			if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j,k+1)] += val;

			grids[Index(i,j,k)] = 0.f;
		}
	}
};

__device__ void atomicVelocityObs_U( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
		{
			if ( grids[Index(i,j,k)] > 0.f )
			{
				if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK )
					grids[Index(i-1,j,k)] = grids[Index(i-1,j,k)] -  grids[Index(i,j,k)];
			}
			else
			{
				if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK )
					grids[Index(i+1,j,k)] = grids[Index(i+1,j,k)] -  grids[Index(i,j,k)];
			}
			grids[Index(i,j,k)] = 0.f;
		}
	}
};

__device__ void atomicVelocityObs_V( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
		{
			if ( grids[Index(i,j,k)] > 0.f )
			{
				if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK )
					grids[Index(i,j-1,k)] = grids[Index(i,j-1,k)] - grids[Index(i,j,k)];
			}
			else
			{
				if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK )
					grids[Index(i,j+1,k)] = grids[Index(i,j+1,k)] - grids[Index(i,j,k)];
			}
			grids[Index(i,j,k)] = 0.f;
		}
	}
};

__device__ void atomicVelocityObs_W( double *grids, cdouble *obstacle )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
		{
			if ( grids[Index(i,j,k)] > 0.f )
			{
				if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK )
					grids[Index(i,j,k-1)] = grids[Index(i,j,k-1)] - grids[Index(i,j,k)];
			}
			else
			{
				if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK )
					grids[Index(i,j,k+1)] = grids[Index(i,j,k+1)] - grids[Index(i,j,k)];
			}
			grids[Index(i,j,k)] = 0.f;
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


__global__ void kernelSumDensity( double *share, cdouble *src, cint no )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );
	share[no] += src[Index(i,j,k)];
};



__global__ void kernelJacobi( double *grid_out, cdouble *grid_in, cdouble diffusion, cdouble divisor )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
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
	}
}

__global__ void kernelGridAdvection( double *grid_out, cdouble *grid_in, cdouble deltatime, cdouble *u_in, cdouble *v_in, cdouble *w_in )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		double u = i - u_in [ Index(i,j,k) ] * deltatime;
		double v = j - v_in [ Index(i,j,k) ] * deltatime;
		double w = k - w_in [ Index(i,j,k) ] * deltatime;
	
		grid_out [ Index(i,j,k) ] = atomicTrilinear ( grid_in, u, v, w, GRIDS_X, GRIDS_Y, GRIDS_Z );
	}
};

__global__ void kernelGradient( double *div, double *p, cdouble *vel_u, cdouble *vel_v, cdouble *vel_w )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{	
		cdouble h = 1.f / GRIDS_X;

		// previous instantaneous magnitude of velocity gradient = (sum of velocity gradients per axis)/2N:
		div [ Index(i,j,k) ] = -0.5f * h * (
				vel_u [ Index(i+1, j, k) ] - vel_u [ Index(i-1, j, k) ] + // gradient of u
				vel_v [ Index(i, j+1, k) ] - vel_v [ Index(i, j-1, k) ] + // gradient of v
				vel_w [ Index(i, j, k+1) ] - vel_w [ Index(i, j, k-1) ]   // gradient of w
			);
		// zero out the present velocity gradient
		p [ Index(i,j,k) ] = 0.f;
	}
};

__global__ void kernelSubtract( double *vel_u, double *vel_v, double *vel_w, cdouble *p )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		// gradient calculated by neighbors

		vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
		vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
		vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );
	}
};

__global__ void kernelAddSource( double *density, double *vel_u, double *vel_v, double *vel_w )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( atomicIsNotHalo( i, j, k, GRIDS_X, GRIDS_Y, GRIDS_Z ) )
	{
		cint half = GRIDS_X / 2;

		if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
		{
			/* add source to grids */
			density[Index(i,j,k)] = DENSITY;

			/* add velocity to grids */
			if ( i < half )
				vel_u[Index(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
			elif( i >= half )
				vel_u[Index(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;

			vel_v[Index(i,j,k)] = VELOCITY;

			if ( k < half )
				vel_w[Index(i,j,k)] = -VELOCITY * DELTATIME * DELTATIME;
			elif ( k >= half )
				vel_w[Index(i,j,k)] =  VELOCITY * DELTATIME * DELTATIME;
		}
	}
};

__global__ void kernelPickData( uchar *c, cdouble *bufs, int ofi, int ofj, int ofk, cint grids )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	ofi = ofi * grids + i;
	ofj = ofj * grids + j;
	ofk = ofk * grids + k;

	/* zero c first */
	c[ ix(ofi,ofj,ofk,VOLUME_X,VOLUME_Y,VOLUME_Z) ] = 0;

	/* append c to volume c */
	int temp = _round( bufs[ Index(i, j, k) ] );
	if ( temp > 0 and temp < 250 )
		c [ ix(ofi,ofj,ofk,VOLUME_X,VOLUME_Y,VOLUME_Z) ] = (uchar) temp;
};

__global__ void kernelInterRootGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	double x = ( pi * GRIDS_X + i ) * rate;
	double y = ( pj * GRIDS_X + j ) * rate;
	double z = ( pk * GRIDS_X + k ) * rate;

	dst[Index(i,j,k)] = atomicTrilinear( src, x, y, z, GRIDS_X, GRIDS_Y, GRIDS_Z );
};

__global__ void kernelInterLeafGrids( double *dst, cdouble *src, cint pi, cint pj, cint pk, cdouble rate )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	int x = _round( ( pi * GRIDS_X + i ) * rate );
	int y = _round( ( pj * GRIDS_X + j ) * rate );
	int z = _round( ( pk * GRIDS_X + k ) * rate );

	dst[Index(x,y,z)] = src[Index(i,j,k)];
};

__global__ void kernelClearHalo( double *grids )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	grids[Index(gst_header,j,k)] = 0.f;
	grids[Index(gst_tailer,j,k)] = 0.f;
	grids[Index(i,gst_header,k)] = 0.f;
	grids[Index(i,gst_tailer,k)] = 0.f;
	grids[Index(i,j,gst_header)] = 0.f;
	grids[Index(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelHandleHalo
	( double *center, cdouble *left, cdouble *right, cdouble *up, cdouble *down, cdouble *front, cdouble *back )
{
	int i, j, k;
	_thread( &i, &j, &k, GRIDS_X, GRIDS_Y, GRIDS_Z );

	center[Index(gst_header,j,k)] = left[Index(gst_tailer,j,k)];
	center[Index(gst_tailer,j,k)] = right[Index(gst_header,j,k)];
	center[Index(i,gst_tailer,k)] = up[Index(i,gst_header,k)];
    center[Index(i,gst_header,k)] = down[Index(i,gst_tailer,k)];
	center[Index(i,j,gst_tailer)] = front[Index(i,j,gst_header)];
	center[Index(i,j,gst_header)] = back[Index(i,j,gst_tailer)];

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