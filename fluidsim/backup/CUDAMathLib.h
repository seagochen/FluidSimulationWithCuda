/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Nov 21, 2013
* <Last Time>     Jan 26, 2014
* <File Name>     CUDAMathLib.h
*/

#ifndef __cuda_math_lib_h_
#define __cuda_math_lib_h_

#include <cuda_runtime.h>

inline __host__ __device__ int sgrand( int *seed )
{
	*seed = (69069 * *seed + 1);
	return *seed;
};

inline __host__ __device__ double sgrandom( int *seed ) 
{
	return ( sgrand( seed ) & 0xffff ) / (double)0x10000;
};

inline __host__ __device__  double sgcrandom( int *seed )
{
	return 2.0 * ( sgrandom( seed ) - 0.5 );
};

inline __host__ __device__ double sginvsqrt ( double x ) 
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
};

inline __host__ __device__ double sgsqrt ( double x )
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1/x;
};

inline __host__ __device__ int sground ( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __host__ __device__ int sgceil ( double x )
{
	int val = sground(x);
	if (x > 0)
	{
		return (val >= (int)x) ? val : (int)x;
	}
	else
	{
		return ((int)x >= val) ? x : val;
	}
};

inline __host__ __device__  int sgfloor(double x)
{
	int val = sground(x);
	if (x > 0)
	{
		return (val < (int)x) ? val : x;
	}
	else
	{
		return ((int)x < val) ? x : val;
	}
};

inline __host__ __device__ int sgfabs(int value)
{
	return (value >= 0) ? value : -value;
};

inline __host__ __device__ double sgfabs(double value)
{
	return (value >= 0.f) ? value : -value;
};

#endif