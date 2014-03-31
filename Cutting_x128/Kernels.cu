/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 23, 2014
* <Last Time>     Mar 30, 2014
* <File Name>     Kernels.cu
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "AtomicFunctions.h"
#include "Kernels.h"


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

	return grid[ ix( x, y, z, tx, ty, tz ) ];
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




// updated: 2014/3/28
__global__ void kernelUpScalingInterpolation( double *dst, cdouble *src, 
						cint srcx, cint srcy, cint srcz,
						cint dstx, cint dsty, cint dstz,
						cint zoomx, cint zoomy, cint zoomz )
{
	int i, j, k;
	_thread(&i, &j, &k, dstx, dsty, dstz);
	
	dst[ix(i,j,k,dstx,dsty,dstz)] = atomicTrilinear( src, 
		(double)i/(double)zoomx,
		(double)j/(double)zoomy,
		(double)k/(double)zoomz,
		srcx, srcy, srcz );
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

		out[ IX(i,j,k) ] = atomicTrilinear( in, velu, velv, velw, tx, ty, tz );
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
		cdouble hx = 1.f / (double)tx;
		cdouble hy = 1.f / (double)ty;
		cdouble hz = 1.f / (double)tz;

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
		u[ IX(i,j,k) ] -= 0.5f * tx * ( prs[ IX(i+1,j,k) ] - prs[ IX(i-1,j,k) ] );
		v[ IX(i,j,k) ] -= 0.5f * ty * ( prs[ IX(i,j+1,k) ] - prs[ IX(i,j-1,k) ] );
		w[ IX(i,j,k) ] -= 0.5f * tz * ( prs[ IX(i,j,k+1) ] - prs[ IX(i,j,k-1) ] );
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


// updated: 2014/3/30
__global__ void kernelSetBound( double *dst, cint dstx, cint dsty, cint dstz )
{
	int i, j, k;
	_thread( &i, &j, &k, dstx, dsty, dstz );

	cint halfx = dstx / 2;
	cint halfz = dstz / 2;

	if ( j < 3 and 
		i >= halfx - 2 and i < halfx + 2 and 
		k >= halfz - 2 and k < halfz + 2 )
	{
		dst[ix(i,j,k, dstx, dsty, dstz)] = MACRO_BOUNDARY_SOURCE;
	}
	else
	{
		dst[ix(i,j,k, dstx, dsty, dstz)] = MACRO_BOUNDARY_BLANK;
	}
};


// updated" 2014/3/30
__global__ void kernelPickData( uchar *volume, cdouble *src, 
		cint dstx, cint dsty, cint dstz,
		cint srcx, cint srcy, cint srcz,
		cint offi, cint offj, cint offk )
{
	int i, j, k;
	_thread( &i, &j, &k, srcx, srcy, srcz );

	volume[ix(i + srcx * offi, j + srcy * offj, k + srcz * offk, dstx, dsty, dstz)] = 
		( src[ix(i, j, k, srcx, srcy, srcz)] > 0.f and 
		src[ix(i, j, k, srcx, srcy, srcz)] < 250.f ) ? (uchar) src[ix(i, j, k, srcx, srcy, srcz)] : 0;
};


// updated: 2014/3/28
__global__ void kernelPickData( uchar *volume, cdouble *src, cint tx, cint ty, cint tz )
{
	int i, j, k;
	_thread(&i, &j, &k, tx, ty, tz);

	volume[ix(i, j, k, tx, ty, tz)] = ( src[ix(i, j, k, tx, ty, tz)] > 0.f and 
		src[ix(i, j, k, tx, ty, tz)] < 250.f ) ? (uchar) src[ix(i, j, k, tx, ty, tz)] : 0;
};


// updated: 2014/3/30
__global__ void kernelAssembleCompBufs( double *dst, cdouble *src,
		cint srcx, cint srcy, cint srcz, 
		cint dstx, cint dsty, cint dstz,
		cint offi, cint offj, cint offk )
{
	int i, j, k;
	_thread( &i, &j, &k, srcx, srcy, srcz );

	int dsti, dstj, dstk;
	dsti = offi * srcx + i;
	dstj = offj * srcy + j;
	dstk = offk * srcz + k;

	if ( dsti < 0 ) dsti = 0;
	if ( dstj < 0 ) dstj = 0;
	if ( dstk < 0 ) dstk = 0;
	if ( dsti >= dstx ) dsti = dstx - 1;
	if ( dstj >= dsty ) dstj = dsty - 1;
	if ( dstk >= dstz ) dstk = dstz - 1;

	dst[ix(dsti, dstj, dstk, dstx, dsty, dstz)] = src[ix(i, j, k, srcx, srcy, srcz)];
};


// updated: 2014/3/30
__global__ void kernelDeassembleCompBufs( double *dst, cdouble *src,
		cint srcx, cint srcy, cint srcz,
		cint dstx, cint dsty, cint dstz,
		cint offi, cint offj, cint offk	)
{
	int i, j, k;
	_thread( &i, &j, &k, dstx, dsty, dstz );

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
};


// updated: 2014/3/30
__global__ void kernelFillBullet( double *dst, cdouble *src,
		cint srcx, cint srcy, cint srcz, 
		cint dstx, cint dsty, cint dstz, 
		cint grdx, cint grdy, cint grdz,
		cint offi, cint offj, cint offk )
{
	int i, j, k;
	_thread( &i, &j, &k, dstx, dsty, dstz );

	dst[ix(i, j, k, dstx, dsty, dstz)] = atomicGetValue( src,
		offi * grdx + i - 1,
		offj * grdy + j - 1,
		offk * grdz + k - 1,
		srcx, srcy, srcz );
};


// updated: 2014/3/31
__global__ void kernelSumDensity
	( double *bufs, cdouble *src, cint no, cint srcx, cint srcy, cint srcz )
{
	int i, j, k;
	_thread( &i, &j, &k, srcx, srcy, srcz );

	bufs[no] += src[ix(i,j,k,srcx,srcy,srcz)]; 
};