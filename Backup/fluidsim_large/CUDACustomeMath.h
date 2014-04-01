/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 23, 2014
* <File Name>     CUDACustomMath.h
*/

#ifndef __cuda_custom_math_h__
#define __cuda_custom_math_h__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

inline __host__ __device__ int atomicRand( int *seed )
{
	*seed = (69069 * *seed + 1);
	return *seed;
};

inline __host__ __device__ double atomicRandom( int *seed ) 
{
	return ( atomicRand( seed ) & 0xffff ) / (double)0x10000;
};

inline __host__ __device__  double atomicCrandom( int *seed )
{
	return 2.0 * ( atomicRandom( seed ) - 0.5 );
};

inline __host__ __device__ double atomicInvsqrt( double x ) 
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
};

inline __host__ __device__ double atomicSqrt( double x )
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1/x;
};

inline __host__ __device__ int atomicRound( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __host__ __device__ int atomicCeil( double x )
{
	int val = atomicRound(x);
	if (x > 0)
	{
		return (val >= (int)x) ? val : (int)x;
	}
	else
	{
		return ((int)x >= val) ? x : val;
	}
};

inline __host__ __device__  int atomicFloor(double x)
{
	int val = atomicRound(x);
	if (x > 0)
	{
		return (val < (int)x) ? val : x;
	}
	else
	{
		return ((int)x < val) ? x : val;
	}
};

inline __host__ __device__ int atomicFabs(int value)
{
	return (value >= 0) ? value : -value;
};

inline __host__ __device__ double atomicFabs(double value)
{
	return (value >= 0.f) ? value : -value;
};


#endif