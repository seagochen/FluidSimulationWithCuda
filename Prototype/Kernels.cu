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

#if 0

#define IX(i,j,k) ix(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X)
#define thread() _thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X);
#define isbound(i,j,k) atomicIXNotHalo(i,j,k,GRIDS_X,GRIDS_X,GRIDS_X)

__global__ void kernelJacobi( double *out, cdouble *in, cdouble diffusion, cdouble divisor )
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

__global__ void kernelAdvection( double *out, cdouble *in, cdouble delta, cdouble *u, cdouble *v, cdouble *w )
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

__global__ void kernelGradient( double *div, double *prs, cdouble *u, cdouble *v, cdouble *w )
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


#undef thread()
#undef isbound(i,j,k)


/************************************************************************************
** etc.                                                                            **
*************************************************************************************/


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

#undef IX(i,j,k)

__global__ void kernelPickData
	( uchar *volume, cdouble *rho, int offi, int offj, int offk, cint gridx, cint gridy, cint gridz )
{
	int i, j, k;
	_thread( &i, &j, &k, gridx, gridy, gridz );

	offi = offi * gridx + i;
	offj = offj * gridy + j;
	offk = offk * gridz + k;

	int dens = _round( rho[ ix(i, j, k, gridx, gridy, gridz) ] );

	volume[ ix( offi, offj, offk, VOLUME_X, VOLUME_Y, VOLUME_Z ) ] = 
		( dens < 250 and dens >= 0 ) ? (uchar)dens : 0;
};

#endif

/************************************************************************************
** Interpolation kernels                                                           **
*************************************************************************************/

// updated: 2014/3/27
__global__ void kernelAssembleCompBufs( double *dst, cint dstx, cint dsty, cint dstz, 
									  cdouble *src, cint srcx, cint srcy, cint srcz,
									  cint offi, cint offj, cint offk, 
									  cdouble zoomx, cdouble zoomy, cdouble zoomz )
{
	int i, j, k;
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int dsti, dstj, dstk;

#if 1
		dsti = offi * srcx + i;
		dstj = offj * srcy + j;
		dstk = offk * srcz + k;
#else
		dsti = _round( (offi * srcx + i) * zoomx );
		dstj = _round( (offj * srcy + j) * zoomy );
		dstk = _round( (offk * srcz + k) * zoomz );
#endif

		if ( dsti < 0 ) dsti = 0;
		if ( dstj < 0 ) dstj = 0;
		if ( dstk < 0 ) dstk = 0;
		if ( dsti >= dstx ) dsti = dstx - 1;
		if ( dstj >= dsty ) dstj = dsty - 1;
		if ( dstk >= dstz ) dstk = dstz - 1;

		dst[ix(dsti, dstj, dstk, dstx, dsty, dstz)] = src[ix(i, j, k, srcx, srcy, srcz)];
}


// updated: 2014/3/27
__global__ void kernelDeassembleCompBufs( double *dst, cint dstx, cint dsty, cint dstz, 
										 cdouble *src, cint srcx, cint srcy, cint srcz,
										 cint offi, cint offj, cint offk, 
										 cdouble zoomx, cdouble zoomy, cdouble zoomz )
{
	int i, j, k;
	_thread( &i, &j, &k, dstx, dsty, dstz );

#if 0
	double srci, srcj, srck;

	srci = ( i + offi * dstx ) * zoomx;
	srcj = ( j + offj * dsty ) * zoomy;
	srck = ( k + offk * dstz ) * zoomz;

	if ( srci < 0 ) srci = 0.f;
	if ( srcj < 0 ) srcj = 0.f;
	if ( srck < 0 ) srck = 0.f;
	if ( srci >= srcx ) srci = srcx - 1.f;
	if ( srcj >= srcy ) srcj = srcy - 1.f;
	if ( srck >= srcz ) srck = srcz - 1.f;

	dst[ix(i, j, k, dstx, dsty, dstz)] = atomicTrilinear( src, srci, srcj, srck, srcx, srcy, srcz );

#else

	int srci, srcj, srck;

	srci = i + offi * dstx;
	srcj = j + offj * dsty;
	srck = k + offk * dstz;

	if ( srci < 0 ) srci = 0;
	if ( srcj < 0 ) srcj = 0;
	if ( srck < 0 ) srck = 0;
	if ( srci >= srcx ) srci = srcx - 1;
	if ( srcj >= srcy ) srcj = srcy - 1;
	if ( srck >= srcz ) srck = dstz - 1;

	dst[ix(i, j, k, dstx, dsty, dstz)] = src[ix(srci, srcj, srck, srcx, srcy, srcz)];

#endif
};

#if 0

#define IX(i,j,k) ix(i, j, k, BULLET_X, BULLET_Y, BULLET_Z )
#define thread() _thread(&i,&j,&k,GRIDS_X,GRIDS_X,GRIDS_X); i++; j++; k++;
#define isbound(i,j,k) atomicIXNotHalo(i, j, k, BULLET_X, BULLET_Y, BULLET_Z )

__global__ void kernelAddSource( double *dens, double *v, cdouble *obst, cdouble dtime, cdouble rate )
{
	int i, j, k;
	thread();

	if ( obst[IX(i,j,k)] < 0 )
	{
		double pop = -obst[IX(i,j,k)] / 100.f;

		/* add source to grids */
		dens[IX(i,j,k)] = DENSITY * rate * dtime * pop;

		v[IX(i,j,k)] = VELOCITY * rate * dtime * pop;
	}
};

#undef IX(i,j,k)
#undef thread()
#undef isbound(i,j,k)

#endif

// updated: 2014/3/27
__global__ void kernelPickData( uchar *volume, cint dstx, cint dsty, cint dstz,
							   cdouble *src, cint srcx, cint srcy, cint srcz,
							   cint offi, cint offj, cint offk, 
							   cdouble zoomx, cdouble zoomy, cdouble zoomz )
{
	int i, j, k;
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int srci, srcj, srck;

	srci = _round(offi * srcx + i * zoomx);
	srcj = _round(offj * srcy + j * zoomy);
	srck = _round(offk * srcz + k * zoomz);

	if ( srci < 0 ) srci = 0;
	if ( srcj < 0 ) srcj = 0;
	if ( srck < 0 ) srck = 0;
	if ( srci >= dstx ) srci = dstx - 1;
	if ( srcj >= dsty ) srcj = dsty - 1;
	if ( srck >= dstz ) srck = dstz - 1;

	volume[ix(srci, srcj, srck, dstx, dsty, dstz)] = ( src[ix(i, j, k, srcx, srcy, srcz)] > 0.f and 
		src[ix(i, j, k, srcx, srcy, srcz)] < 250.f ) ? (uchar) src[ix(i, j, k, srcx, srcy, srcz)] : 0;
};


#define thread() \
	int i, j, k; \
	_thread( &i, &j, &k, tx, ty, tz); \
	i++; j++; k++;

#define isbound() \
	atomicIXNotHalo( i, j, k, tx, ty, tz )

#define IX(i,j,k) \
	ix(i, j, k, tx, ty, tz )

// updated: 2014/3/27
__global__ void kernelJacobi( double *out, cdouble *in, 
							 cint tx, cint ty, cint tz,
							 cdouble diffusion, cdouble divisor )
{
	thread();

	if ( isbound() )
	{
		double dix = ( divisor > 0 ) ? divisor : 1.f;

		out[ IX(i,j,k) ] = ( in[ IX(i,j,k) ] + diffusion * (
			out[ IX(i-1,j,k) ] + out[ IX(i+1,j,k) ] +
			out[ IX(i,j-1,k) ] + out[ IX(i,j+1,k) ] +
			out[ IX(i,j,k-1) ] + out[ IX(i,j,k+1) ]
			) ) / dix;
	}
};

// updated: 2014/3/27
__global__ void kernelAdvection( double *out, cdouble *in, 
								cint tx, cint ty, cint tz,
								cdouble delta, cdouble *u, cdouble *v, cdouble *w )
{
	thread();

	if ( isbound( i, j, k ) )
	{
		double velu = i - u[ IX(i,j,k) ] * delta;
		double velv = j - v[ IX(i,j,k) ] * delta;
		double velw = k - w[ IX(i,j,k) ] * delta;

		out[ IX(i,j,k) ] = atomicTrilinear( in, velu, velv, velw, BULLET_X, BULLET_Y, BULLET_Z );
	}
};

// updated: 2014/3/27
__global__ void kernelGradient( double *div, double *prs,
							   cint tx, cint ty, cint tz,
							   cdouble *u, cdouble *v, cdouble *w )
{
	thread();

	if ( isbound( i, j, k ) )
	{
		cdouble hx = 1.f / (double)BULLET_X;
		cdouble hy = 1.f / (double)BULLET_Y;
		cdouble hz = 1.f / (double)BULLET_Z;

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

// updated: 2014/3/27
__global__ void kernelSubtract( double *u, double *v, double *w, double *prs,
							   cint tx, cint ty, cint tz )
{
	thread();

	if ( isbound( i, j, k ) )
	{
		u[ IX(i,j,k) ] -= 0.5f * BULLET_X * ( prs[ IX(i+1,j,k) ] - prs[ IX(i-1,j,k) ] );
		v[ IX(i,j,k) ] -= 0.5f * BULLET_Y * ( prs[ IX(i,j+1,k) ] - prs[ IX(i,j-1,k) ] );
		w[ IX(i,j,k) ] -= 0.5f * BULLET_Z * ( prs[ IX(i,j,k+1) ] - prs[ IX(i,j,k-1) ] );
	}
};

// updated: 2014/3/27
__global__ void kernelAddSource( double *dens, double *v,
								cint tx, cint ty, cint tz,
								cdouble *obst, cdouble dtime, cdouble rate )
{
	thread();

	if ( obst[IX(i,j,k)] < 0 )
	{
		double pop = -obst[IX(i,j,k)] / 100.f;

		/* add source to grids */
		dens[IX(i,j,k)] = DENSITY * rate * dtime * pop;

		v[IX(i,j,k)] = VELOCITY * rate * dtime * pop;
	}
};

#undef IX(i,j,k)
#undef isbound()
#undef thread()