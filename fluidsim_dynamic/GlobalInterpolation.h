/**
* <Author>      Orlando Chen
* <First>       Nov 21, 2013
* <Last>		Jan 26, 2014
* <File>        GlobalInterpolation.h
*/

#ifndef __global_interpolation_h_
#define __global_interpolation_h_

#include "DataStructures.h"
#include "Parameter.h"
#include <cuda_runtime_api.h>


#define USE_DEFAULT     0
#define USE_UP_GRID     1
#define USE_DOWN_GRID   2
#define USE_LEFT_GRID   3
#define USE_RIGHT_GRID  4
#define USE_FRONT_GRID  5
#define USE_BACK_GRID   6

__device__ 
double at_cell (double const *grid, int const x, int const y, int const z)
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_tailer ) return 0.f;
	if ( y > gst_tailer ) return 0.f;
	if ( z > gst_tailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

__device__
double at_global ( double const *grid, double const *extgrid, int x, int y, int z, int const FACE )
{
	switch ( FACE )
	{
	case USE_UP_GRID:
		y = y - GRIDS_X;
		return at_cell( extgrid, x, y, z );

	case USE_DOWN_GRID:
		y = y + GRIDS_X;
		return at_cell( extgrid, x, y, z );
		
	case USE_LEFT_GRID:
		x = x + GRIDS_X;
		return at_cell( extgrid, x, y, z );

	case USE_RIGHT_GRID:
		x = x - GRIDS_X;
		return at_cell( extgrid, x, y, z );

	case USE_FRONT_GRID:
		z = z - GRIDS_X;
		return at_cell( extgrid, x, y, z );

	case USE_BACK_GRID:
		z = z + GRIDS_X;
		return at_cell( extgrid, x, y, z );

	default:
		break;
	}

	return at_cell( grid, x, y, z );
};

__device__  
void local_vertices( double *stores,
	double const *grid, double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	stores[ 0 ] = at_cell( grid, i, j, k );      // v000
	stores[ 1 ] = at_cell( grid, i, j+1, k );    // v001
	stores[ 2 ] = at_cell( grid, i, j+1, k+1 );  // v011
	stores[ 3 ] = at_cell( grid, i, j, k+1 );    // v010
	stores[ 4 ] = at_cell( grid, i+1, j, k );    // v100
	stores[ 5 ] = at_cell( grid, i+1, j+1, k );  // v101
	stores[ 6 ] = at_cell( grid, i+1, j+1, k+1 );// v111
	stores[ 7 ] = at_cell( grid, i+1, j, k+1 );  // v110
}

__device__
double local_trilinear( double *stores,
	double const *grid, double const x, double const y, double const z )
{
	local_vertices( stores, grid, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = stores[ 0 ] * ( 1 - dx ) + stores[ 1 ] * dx;
	double c10 = stores[ 3 ] * ( 1 - dx ) + stores[ 2 ] * dx;
	double c01 = stores[ 4 ] * ( 1 - dx ) + stores[ 5 ] * dx;
	double c11 = stores[ 7 ] * ( 1 - dx ) + stores[ 6 ] * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

__device__
double global_point(
	double const *grid, int const x, int const y, int const z,
	double const *up,    double const *down, 
	double const *left,  double const *right,
	double const *front, double const *back	)
{
	if ( x >= gst_header and x < gst_tailer )
	{
		if ( y >= gst_header and y < gst_tailer )
		{
			if ( z >= gst_header and z < gst_tailer )
			{
				return at_global( grid, grid, x, y, z, USE_DEFAULT );
			}
			elif ( z < gst_header )
			{
				return at_global( grid, back, x, y, z, USE_BACK_GRID );
			}
			else
			{
				return at_global( grid, front, x, y, z, USE_FRONT_GRID );
			}
		}
		elif ( y < gst_header and
			z >= gst_header and z < gst_tailer )
		{
			return at_global( grid, down, x, y, z, USE_DOWN_GRID );
		}
		elif ( y >= gst_tailer and
			z >= gst_header and z < gst_tailer )
		{
			return at_global( grid, up, x, y, z, USE_UP_GRID );
		}
	}
	elif ( x >= gst_tailer and 
		y >= gst_header and y < gst_tailer and z >= gst_header and z < gst_tailer )
	{
		return at_global( grid, right, x, y, z, USE_RIGHT_GRID );
	}
	elif ( x < gst_header and
		y >= gst_header and y < gst_tailer and z >= gst_header and z < gst_tailer )
	{
		return at_global( grid, left, x, y, z, USE_LEFT_GRID );
	}
};

__device__
void global_vertices( double *stores, 
	double const *grid,
	double const *up,    double const *down, 
	double const *left,  double const *right,
	double const *front, double const *back,
	double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	stores[ 0 ] = global_point( grid, i, j, k, up, down, left, right, front, back );      // v000
	stores[ 1 ] = global_point( grid, i, j+1, k, up, down, left, right, front, back );    // v001
	stores[ 2 ] = global_point( grid, i, j+1, k+1, up, down, left, right, front, back );  // v011
	stores[ 3 ] = global_point( grid, i, j, k+1, up, down, left, right, front, back );    // v010
	stores[ 4 ] = global_point( grid, i+1, j, k, up, down, left, right, front, back );    // v100
	stores[ 5 ] = global_point( grid, i+1, j+1, k, up, down, left, right, front, back );  // v101
	stores[ 6 ] = global_point( grid, i+1, j+1, k+1, up, down, left, right, front, back );// v111
	stores[ 7 ] = global_point( grid, i+1, j, k+1, up, down, left, right, front, back );  // v110

};

__device__
double global_trilinear ( double *stores,
	double const *grid,
	double const *up,    double const *down, 
	double const *left,  double const *right,
	double const *front, double const *back,
	double const x, double const y, double const z )
{
	global_vertices( stores, grid, up, down, left, right, front, back, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = stores[ 0 ] * ( 1 - dx ) + stores[ 1 ] * dx;
	double c10 = stores[ 3 ] * ( 1 - dx ) + stores[ 2 ] * dx;
	double c01 = stores[ 4 ] * ( 1 - dx ) + stores[ 5 ] * dx;
	double c11 = stores[ 7 ] * ( 1 - dx ) + stores[ 6 ] * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

#endif