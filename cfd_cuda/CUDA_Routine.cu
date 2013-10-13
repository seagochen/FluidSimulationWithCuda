/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <First>       Oct 12, 2013
* <Last>		Oct 12, 2013
* <File>        CUDA_Routine.cpp
*/

#include "Macro_Definitions.h"

#define _CUDA_ROUTINE_CPP_

#if GPU_ON

#include <SGE\SGUtils.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void add_source ( float *grid, float *src )
{
	int i, size=(GRIDS_WITHOUT_GHOST+2)*(GRIDS_WITHOUT_GHOST+2);
	for ( i=0 ; i<size ; i++ ) grid[i] += DELTA_TIME*src[i];
}

void set_bnd ( float *grid, int boundary )
{
	int i;

	for ( i=1 ; i<=GRIDS_WITHOUT_GHOST ; i++ ) {
		grid[Index(0  ,i)] = boundary==1 ? -grid[Index(1,i)] : grid[Index(1,i)];
		grid[Index(GRIDS_WITHOUT_GHOST+1,i)] = boundary==1 ? -grid[Index(GRIDS_WITHOUT_GHOST,i)] : grid[Index(GRIDS_WITHOUT_GHOST,i)];
		grid[Index(i,0  )] = boundary==2 ? -grid[Index(i,1)] : grid[Index(i,1)];
		grid[Index(i,GRIDS_WITHOUT_GHOST+1)] = boundary==2 ? -grid[Index(i,GRIDS_WITHOUT_GHOST)] : grid[Index(i,GRIDS_WITHOUT_GHOST)];
	}
	grid[Index(0  ,0  )] = 0.5f*(grid[Index(1,0  )]+grid[Index(0  ,1)]);
	grid[Index(0  ,GRIDS_WITHOUT_GHOST+1)] = 0.5f*(grid[Index(1,GRIDS_WITHOUT_GHOST+1)]+grid[Index(0  ,GRIDS_WITHOUT_GHOST)]);
	grid[Index(GRIDS_WITHOUT_GHOST+1,0  )] = 0.5f*(grid[Index(GRIDS_WITHOUT_GHOST,0  )]+grid[Index(GRIDS_WITHOUT_GHOST+1,1)]);
	grid[Index(GRIDS_WITHOUT_GHOST+1,GRIDS_WITHOUT_GHOST+1)] = 0.5f*(grid[Index(GRIDS_WITHOUT_GHOST,GRIDS_WITHOUT_GHOST+1)]+grid[Index(GRIDS_WITHOUT_GHOST+1,GRIDS_WITHOUT_GHOST)]);
}


void lin_solve (float *grid, float *grid0, int boundary, float a, float c )
{
	int i, j, k;

	for ( k=0 ; k<20 ; k++ ) 
	{
		for ( i=1 ; i<=GRIDS_WITHOUT_GHOST ; i++ )
		{
			for ( j=1 ; j<=GRIDS_WITHOUT_GHOST ; j++ ) 
			{
				grid[Index(i,j)] = (grid0[Index(i,j)] + a*(grid[Index(i-1,j)]+grid[Index(i+1,j)]+grid[Index(i,j-1)]+grid[Index(i,j+1)]))/c;
			}
		}
		set_bnd ( grid, boundary );
	}
}


void diffuse ( float *grid, float *grid0, int boundary, float diff )
{
	float a=DELTA_TIME*diff*GRIDS_WITHOUT_GHOST*GRIDS_WITHOUT_GHOST;
	lin_solve ( grid, grid0, boundary, a, 1+4*a );
}


void advect ( float *density, float *density0, float *u, float *v,  int boundary )
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	dt0 = DELTA_TIME*GRIDS_WITHOUT_GHOST;
	for ( i=1 ; i<=GRIDS_WITHOUT_GHOST ; i++ ) 
	{
		for ( j=1 ; j<=GRIDS_WITHOUT_GHOST ; j++ ) 
		{
			x = i-dt0*u[Index(i,j)]; y = j-dt0*v[Index(i,j)];
			if (x<0.5f) x=0.5f; if (x>GRIDS_WITHOUT_GHOST+0.5f) x=GRIDS_WITHOUT_GHOST+0.5f; i0=(int)x; i1=i0+1;
			if (y<0.5f) y=0.5f; if (y>GRIDS_WITHOUT_GHOST+0.5f) y=GRIDS_WITHOUT_GHOST+0.5f; j0=(int)y; j1=j0+1;
			s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
			density[Index(i,j)] = s0*(t0*density0[Index(i0,j0)]+t1*density0[Index(i0,j1)])+
				s1*(t0*density0[Index(i1,j0)]+t1*density0[Index(i1,j1)]);
		}
	}
	set_bnd ( density, boundary );
}


void project ( float *u, float *v, float *u0, float *v0 )
{
	int i, j;

	for ( i=1 ; i<=GRIDS_WITHOUT_GHOST ; i++ )
	{
		for ( j=1 ; j<=GRIDS_WITHOUT_GHOST ; j++ )
		{
			v0[Index(i,j)] = -0.5f*(u[Index(i+1,j)]-u[Index(i-1,j)]+v[Index(i,j+1)]-v[Index(i,j-1)])/GRIDS_WITHOUT_GHOST;		
			u0[Index(i,j)] = 0;
		}
	}	
	set_bnd ( v0, 0 ); set_bnd ( u0, 0 );

	lin_solve ( u0, v0, 0, 1, 4 );

	for ( i=1 ; i<=GRIDS_WITHOUT_GHOST ; i++ )
	{
		for ( j=1 ; j<=GRIDS_WITHOUT_GHOST ; j++ ) 
		{
			u[Index(i,j)] -= 0.5f*GRIDS_WITHOUT_GHOST*(u0[Index(i+1,j)]-u0[Index(i-1,j)]);
			v[Index(i,j)] -= 0.5f*GRIDS_WITHOUT_GHOST*(u0[Index(i,j+1)]-u0[Index(i,j-1)]);
		}
	}
	set_bnd ( u, 1 ); set_bnd ( v, 2 );
}


void dens_step ( float *grid, float *grid0, float *u, float *v )
{
	add_source ( grid, grid0 );
	SWAP ( grid0, grid ); diffuse ( grid, grid0, 0, DIFFUSION );
	SWAP ( grid0, grid ); advect ( grid, grid0, u, v, 0 );
}


void vel_step ( float *u, float *v, float *u0, float *v0 )
{
	add_source ( u, u0 ); add_source ( v, v0 );
	SWAP ( u0, u ); diffuse ( u, u0, 1, VISCOSITY );
	SWAP ( v0, v ); diffuse ( v, v0, 2, VISCOSITY );
	project ( u, v, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );
	advect ( u, u0, u0, v0, 1 ); advect ( v, v0, u0, v0, 2 );
	project ( u, v, u0, v0 );
}

#endif