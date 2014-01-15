/**
* <Author>      Orlando Chen
* <First>       Jan 09, 2014
* <Last>		Jan 09, 2014
* <File>        FluidMathLibDynamic.h
*/

#ifndef __my_math_dynamic_h_
#define __my_math_dynamic_h_

#pragma once

#include <cuda_runtime.h>
#include "FluidSimAreaDynamic.h"

inline __host__ __device__ int sgrand( int *seed )
{
	*seed = ( 69069 * *seed + 1 );
	return *seed;
};

inline __host__ __device__ double sgrandom( int *seed ) 
{
	return ( sgrand ( seed ) & 0xffff ) / (double)0x10000;
};

inline __host__ __device__  double sgcrandom( int *seed )
{
	return 2.0 * ( sgrandom ( seed ) - 0.5 );
};

inline __host__ __device__ double sginvsqrt ( double x ) 
{
	double xhalf = 0.5f * x;
	int i = *( int* )&x;
	i = 0x5f3759df - ( i >> 1 );
	x = *( double* )&i;
	x = x * ( 1.5f - xhalf * x * x );
	return x;
};

inline __host__ __device__ double sgsqrt ( double x )
{
	double xhalf = 0.5f*x;
	int i = *( int* )&x;
	i = 0x5f3759df - ( i >> 1 );
	x = *( double* )&i;
	x = x * ( 1.5f - xhalf * x * x );
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

inline __host__ __device__  double atCell (double const *grid, int const x, int const y, int const z)
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_trailer ) return 0.f;
	if ( y > gst_trailer ) return 0.f;
	if ( z > gst_trailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

inline __host__ __device__  void vertices (
	double *c000, double *c001, double *c011, double *c010,
	double *c100, double *c101, double *c111, double *c110,
	double const *grid, double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	*c000 = atCell ( grid, i, j, k );
	*c001 = atCell ( grid, i, j+1, k );
	*c011 = atCell ( grid, i, j+1, k+1 );
	*c010 = atCell ( grid, i, j, k+1 );
	*c100 = atCell ( grid, i+1, j, k );
	*c101 = atCell ( grid, i+1, j+1, k );
	*c111 = atCell ( grid, i+1, j+1, k+1 );
	*c110 = atCell ( grid, i+1, j, k+1 );
}

inline __host__ __device__  double trilinear ( double const *grid, double const x, double const y, double const z )
{
	double v000, v001, v010, v011, v100, v101, v110, v111;
	vertices ( &v000, &v001, &v011, &v010,
		&v100, &v101, &v111, &v110,
		grid, x, y, z );

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

#endif