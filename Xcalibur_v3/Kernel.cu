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

#include "MacroDefinition.h"

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

inline __device__ bool atomicIXNotHalo
	( cint i, cint j, cint k, cint tx, cint ty, cint tz )
{
	if ( i eqt 0 or i eqt tx - 1 ) return false;
	if ( j eqt 0 or j eqt ty - 1 ) return false;
	if ( k eqt 0 or k eqt tz - 1 ) return false;

	return true;
};


/************************************************************************************
** The following is auxiliary kernels only.                                        **
*************************************************************************************/


__global__ void kernelSumDensity
	( double *share, cdouble *src, cint no, cint tx, cint ty, cint tz )
{
	int i, j, k;
	_thread( &i, &j, &k, tx, ty, tz );
	share[no] += src[ix(i,j,k,tx,ty,tz)];
};



/************************************************************************************
** Basic kernels for solving Navier-Stokes equation.                               **
*************************************************************************************/


#define IX(i,j,k) ix(i,j,k,tx,ty,tz)
#define thread() _thread(&i,&j,&k,GRIDS_X,GRIDS_Y,GRIDS_Z); i+=1;j+=1;k+=1;

__global__ void kernelJacobi
	( double *out, cdouble *in, cdouble diffusion, cdouble divisor,
	cint tx, cint ty, cint tz )
{
	int i, j, k;
	thread();

	if ( atomicIXNotHalo( i, j, k, tx, ty, tz ) )
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
	cdouble *u, cdouble *v, cdouble *w,
	cint tx, cint ty, cint tz )
{
	int i, j, k;
	thread();

	if ( atomicIXNotHalo( i, j, k, tx, ty, tz ) )
	{
		double velu = i - u[ IX(i,j,k) ] * delta;
		double velv = j - v[ IX(i,j,k) ] * delta;
		double velw = k - w[ IX(i,j,k) ] * delta;

		out[ IX(i,j,k) ] = atomicTrilinear( in, velu, velv, velw, tx, ty, tz );
	}
};

__global__ void kernelGradient( double *div, double *prs,
							   cdouble *u, cdouble *v, cdouble *w,
							   cint tx, cint ty, cint tz )
{
	int i, j, k;
	thread();

	if ( atomicIXNotHalo( i, j, k, tx, ty, tz ) )
	{
		cdouble hx = 1.f / (double)tx;
		cdouble hy = 1.f / (double)ty;
		cdouble hz = 1.f / (double)tz;

		div[ IX(i,j,k) ] = -0.5f * (
			hx * ( u[ IX(i+1,j,k) ] - u[ IX(i-1,j,k) ] ) +
			hy * ( v[ IX(i,j+1,k) ] - v[ IX(i,j-1,k) ] ) +
			hz * ( w[ IX(i,j,k+1) ] - w[ IX(i,j,k-1) ] ) );

		prs[ IX(i,j,k) ] = 0.f;
	}
};

__global__ void kernelSubtract( double *u, double *v, double *w, double *prs, 
							  cint tx, cint ty, cint tz )
{
	int i, j, k;
	thread();

	if ( atomicIXNotHalo( i, j, k, tx, ty, tz ) )
	{
		u[ IX(i,j,k) ] -= 0.5f * tx * ( prs[ IX(i+1,j,k) ] - prs[ IX(i-1,j,k) ] );
		v[ IX(i,j,k) ] -= 0.5f * ty * ( prs[ IX(i,j+1,k) ] - prs[ IX(i,j-1,k) ] );
		w[ IX(i,j,k) ] -= 0.5f * tz * ( prs[ IX(i,j,k+1) ] - prs[ IX(i,j,k-1) ] );
	}
};


/************************************************************************************
** Adding density and velocity source for fluid simulation                         **
*************************************************************************************/

__global__ void kernelAddSource
	( double *den, double *u, double *v, double *w, 
	cdouble *obst, cdouble rho, cdouble vel, cdouble delta, cint time,
	cint tx, cint ty, cint tz )
{
	int i, j, k;
	thread();

	if ( atomicIXNotHalo( i, j, k, tx, ty, tz ) )
	{
		if ( obst[ IX(i,j,k) ] < 0 )
		{
			double rate = -obst[ IX(i,j,k) ] / 100.f;
			double randno = _random( _rand(time) );

			/* add rho to density field */
			den[ IX(i,j,k) ] = rate * randno * rho * delta;

			/* add velocity to velocity field */
			v[ IX(i,j,k) ] = rate * vel * delta;
						
			if ( randno < 0.25f and randno >= 0.f )
			{
				u[ IX(i,j,k) ] = -rate * vel * delta * delta;
				w[ IX(i,j,k) ] = -rate * vel * delta * delta;
			}
			elif ( randno >= 0.25f and randno < 0.5f )
			{
				u[ IX(i,j,k) ] = -rate * vel * delta * delta;
				w[ IX(i,j,k) ] =  rate * vel * delta * delta;				
			}
			elif ( randno >= 0.5f and randno < 0.75f )
			{
				u[ IX(i,j,k) ] =  rate * vel * delta * delta;
				w[ IX(i,j,k) ] = -rate * vel * delta * delta;
			}
			else
			{
				u[ IX(i,j,k) ] = rate * vel * delta * delta;
				w[ IX(i,j,k) ] = rate * vel * delta * delta;
			}
		}
	}
};

#undef thread()

/************************************************************************************
** Data transform & root to leaf, leaf to root                                     **
*************************************************************************************/

__global__ void kernelDensToVolume( uchar *volume, cdouble *rho, cint offi, cint offj, cint offk,
								   cint tx, cint ty, cint tz, cint vx, cint vy, cint vz )
{
	int i, j, k;
	_thread( &i, &j, &k, tx, ty, tz );

	int ofi = offi * tx + i;
	int ofj = offj * ty + j;
	int ofk = offk * tz + k;

	/* zero c first */
	volume[ ix(ofi, ofj, ofk, vx, vy, vz) ] = 0;

	/* append c to volume c */
	uchar temp = _round( rho[ ix(i, j, k, tx, ty, tz) ] );
	if ( temp > 0 and temp < 250 )
		volume [ ix(ofi, ofj, ofk, vx, vy, vz) ] = temp;
};


__global__ void kernelDataFromRoot( double *dst, cdouble *src, 
								   cdouble offx, cdouble offy, cdouble offz, cdouble scale,
								   cint tx, cint ty, cint tz )
{
	int i, j, k;
	_thread( &i, &j, &k, tx, ty, tz );

	double x = ( offx * tx + i ) * scale;
	double y = ( offy * ty + j ) * scale;
	double z = ( offz * tz + k ) * scale;

	dst[ IX(i,j,k) ] = atomicTrilinear( src, x, y, z, tx, ty, tz );
};


__global__ void kernelDataFromLeaf( double *dst, cdouble *src, 
								   cdouble offx, cdouble offy, cdouble offz, cdouble scale,
								   cint tx, cint ty, cint tz )
{
	int i, j, k;
	_thread( &i, &j, &k, tx, ty, tz );

	int x = _round( ( offx * tx + i ) * scale );
	int y = _round( ( offy * ty + j ) * scale );
	int z = _round( ( offz * tz + k ) * scale );

	dst[ IX(x,y,z) ] = src[ IX(i,j,k) ];
};

#undef IX(i,j,k)



#define IXt(i,j,k) ix(i,j,k,tx,ty,tz)
#define IXb(i,j,k) ix(i,j,k,bx,by,bz)

__global__ void kernelHandleHalo( double *bullet, 
								 cdouble *left,  cdouble *right,
								 cdouble *up,    cdouble *down,
								 cdouble *front, cdouble *back,
								 cint tx, cint ty, cint tz, cint bx, cint by, cint bz )
{
	int i, j, k;
	_thread( &i, &j, &k, tx, ty, tz );

	bullet[ IXb( 0,    j+1, k+1 ) ] = left [ IXt( tx-1, j, k ) ];
	bullet[ IXb( bx-1, j+1, k+1 ) ] = right[ IXt( 0,    j, k ) ];
	bullet[ IXb( i+1, by-1, k+1 ) ] = up   [ IXt( i, 0,    k ) ];
	bullet[ IXb( i+1,    0, k+1 ) ] = down [ IXt( i, ty-1, k ) ];
	bullet[ IXb( i+1, j+1,  0   ) ] = back [ IXt( i, j, tz-1 ) ];
	bullet[ IXb( i+1, j+1, bz-1 ) ] = front[ IXt( i, j,    0 ) ];

};

#undef IXt(i,j,k)
#undef IXb(i,j,k)