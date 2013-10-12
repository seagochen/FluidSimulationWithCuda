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

#define _CUDA_ROUTINE_CPP_

#include "Macro_Definitions.h"

#if !GPU_ON

void add_source ( int GridSize, float * grid, float * src, float dt )
{
	int i, size=(GridSize+2)*(GridSize+2);
	for ( i=0 ; i<size ; i++ ) grid[i] += dt*src[i];
}

void set_bnd ( int GridSize, int boundary, float * grid )
{
	int i;

	for ( i=1 ; i<=GridSize ; i++ ) {
		grid[Index(0  ,i)] = boundary==1 ? -grid[Index(1,i)] : grid[Index(1,i)];
		grid[Index(GridSize+1,i)] = boundary==1 ? -grid[Index(GridSize,i)] : grid[Index(GridSize,i)];
		grid[Index(i,0  )] = boundary==2 ? -grid[Index(i,1)] : grid[Index(i,1)];
		grid[Index(i,GridSize+1)] = boundary==2 ? -grid[Index(i,GridSize)] : grid[Index(i,GridSize)];
	}
	grid[Index(0  ,0  )] = 0.5f*(grid[Index(1,0  )]+grid[Index(0  ,1)]);
	grid[Index(0  ,GridSize+1)] = 0.5f*(grid[Index(1,GridSize+1)]+grid[Index(0  ,GridSize)]);
	grid[Index(GridSize+1,0  )] = 0.5f*(grid[Index(GridSize,0  )]+grid[Index(GridSize+1,1)]);
	grid[Index(GridSize+1,GridSize+1)] = 0.5f*(grid[Index(GridSize,GridSize+1)]+grid[Index(GridSize+1,GridSize)]);
}


void lin_solve ( int GridSize, int boundary, float * grid, float * grid0, float a, float c )
{
	int i, j, k;

	for ( k=0 ; k<20 ; k++ ) 
	{
		for ( i=1 ; i<=GridSize ; i++ )
		{
			for ( j=1 ; j<=GridSize ; j++ ) 
			{
				grid[Index(i,j)] = (grid0[Index(i,j)] + a*(grid[Index(i-1,j)]+grid[Index(i+1,j)]+grid[Index(i,j-1)]+grid[Index(i,j+1)]))/c;
			}
		}
		set_bnd ( GridSize, boundary, grid );
	}
}


void diffuse ( int GridSize, int boundary, float * grid, float * grid0, float diff, float dt )
{
	float a=dt*diff*GridSize*GridSize;
	lin_solve ( GridSize, boundary, grid, grid0, a, 1+4*a );
}


void advect ( int GridSize, int boundary, float * density, float * density0, float * u, float * v, float dt )
{
	int i, j, i0, j0, i1, j1;
	float grid, y, s0, t0, s1, t1, dt0;

	dt0 = dt*GridSize;
	for ( i=1 ; i<=GridSize ; i++ ) 
	{
		for ( j=1 ; j<=GridSize ; j++ ) 
		{
			grid = i-dt0*u[Index(i,j)]; y = j-dt0*v[Index(i,j)];
			if (grid<0.5f) grid=0.5f; if (grid>GridSize+0.5f) grid=GridSize+0.5f; i0=(int)grid; i1=i0+1;
			if (y<0.5f) y=0.5f; if (y>GridSize+0.5f) y=GridSize+0.5f; j0=(int)y; j1=j0+1;
			s1 = grid-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
			density[Index(i,j)] = s0*(t0*density0[Index(i0,j0)]+t1*density0[Index(i0,j1)])+
				s1*(t0*density0[Index(i1,j0)]+t1*density0[Index(i1,j1)]);
		}
	}
	set_bnd ( GridSize, boundary, density );
}


void project ( int GridSize, float * u, float * v, float * p, float * div )
{
	int i, j;

	for ( i=1 ; i<=GridSize ; i++ )
	{
		for ( j=1 ; j<=GridSize ; j++ )
		{
			div[Index(i,j)] = -0.5f*(u[Index(i+1,j)]-u[Index(i-1,j)]+v[Index(i,j+1)]-v[Index(i,j-1)])/GridSize;		
			p[Index(i,j)] = 0;
		}
	}	
	set_bnd ( GridSize, 0, div ); set_bnd ( GridSize, 0, p );

	lin_solve ( GridSize, 0, p, div, 1, 4 );

	for ( i=1 ; i<=GridSize ; i++ )
	{
		for ( j=1 ; j<=GridSize ; j++ ) 
		{
			u[Index(i,j)] -= 0.5f*GridSize*(p[Index(i+1,j)]-p[Index(i-1,j)]);
			v[Index(i,j)] -= 0.5f*GridSize*(p[Index(i,j+1)]-p[Index(i,j-1)]);
		}
	}
	set_bnd ( GridSize, 1, u ); set_bnd ( GridSize, 2, v );
}


void dens_step ( float * grid, float * grid0, float * u, float * v )
{
	add_source ( GridSize, grid, grid0, dt );
	SWAP ( grid0, grid ); diffuse ( GridSize, 0, grid, grid0, diff, dt );
	SWAP ( grid0, grid ); advect ( GridSize, 0, grid, grid0, u, v, dt );
}


void vel_step ( float * u, float * v, float * u0, float * v0 )
{
	add_source ( GridSize, u, u0, dt ); add_source ( GridSize, v, v0, dt );
	SWAP ( u0, u ); diffuse ( GridSize, 1, u, u0, visc, dt );
	SWAP ( v0, v ); diffuse ( GridSize, 2, v, v0, visc, dt );
	project ( GridSize, u, v, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );
	advect ( GridSize, 1, u, u0, u0, v0, dt ); advect ( GridSize, 2, v, v0, u0, v0, dt );
	project ( GridSize, u, v, u0, v0 );
}

#endif