#ifndef __my_math_h_
#define __my_math_h_

#pragma once

#include <cuda_runtime.h>
#include "helper_math.h"
#include "fluidsim.h"

inline __host__ __device__ int SG_rand( int *seed )
{
	*seed = (69069 * *seed + 1);
	return *seed;
};

inline __host__ __device__ float SG_random( int *seed ) 
{
	return ( SG_rand( seed ) & 0xffff ) / (float)0x10000;
};

inline __host__ __device__  float SG_crandom( int *seed )
{
	return 2.0 * ( SG_random( seed ) - 0.5 );
};

inline __host__ __device__ float SG_invsqrt ( float x ) 
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(float*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
};

inline __host__ __device__ float SG_sqrt ( float x )
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(float*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1/x;
};

inline __host__ __device__ int SG_round ( float x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __host__ __device__ int SG_ceil ( float x )
{
	int val = SG_round(x);
	if (x > 0)
	{
		return (val >= (int)x) ? val : (int)x;
	}
	else
	{
		return ((int)x >= val) ? x : val;
	}
};

inline __host__ __device__  int SG_floor(float x)
{
	int val = SG_round(x);
	if (x > 0)
	{
		return (val < (int)x) ? val : x;
	}
	else
	{
		return ((int)x < val) ? x : val;
	}
};

#endif