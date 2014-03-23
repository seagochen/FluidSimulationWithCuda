#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"

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

__global__ void kernelAddSource
	( double *den, double *u, double *v, double *w, 
	cdouble *obst, cdouble rho, cdouble vel, cdouble delta, cint time )
{
	int i, j, k;
	thread();

	if ( isbound( i, j, k ) )
	{
		if ( obst[ IX(i,j,k) ] < 0 )
		{
			double rate = -obst[ IX(i,j,k) ] / 100.f;
//			double rate = 1.f;
//			double randno = _random( _rand(time) );

			/* add rho to density field */
//			den[ IX(i,j,k) ] = rate * randno * rho * delta;
			den[ IX(i,j,k) ] = rate * rho * delta;

			/* add velocity to velocity field */
			v[ IX(i,j,k) ] = rate * vel * delta;
						
//			if ( randno < 0.25f and randno >= 0.f )
//			{
//				u[ IX(i,j,k) ] = -rate * vel * delta * delta;
//				w[ IX(i,j,k) ] = -rate * vel * delta * delta;
//			}
//			elif ( randno >= 0.25f and randno < 0.5f )
//			{
//				u[ IX(i,j,k) ] = -rate * vel * delta * delta;
//				w[ IX(i,j,k) ] =  rate * vel * delta * delta;				
//			}
//			elif ( randno >= 0.5f and randno < 0.75f )
//			{
//				u[ IX(i,j,k) ] =  rate * vel * delta * delta;
//				w[ IX(i,j,k) ] = -rate * vel * delta * delta;
//			}
//			else
//			{
//				u[ IX(i,j,k) ] = rate * vel * delta * delta;
//				w[ IX(i,j,k) ] = rate * vel * delta * delta;
//			}
		}
	}
};


#undef IX(i,j,k)
#undef thread()
#undef isbound(i,j,k)