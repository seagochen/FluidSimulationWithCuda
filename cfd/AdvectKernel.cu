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
* <First>       Nov 25, 2013
* <Last>		Nov 25, 2013
* <File>        AdvectKernel.cu
*/

#ifndef __advect_kernel_cu_
#define __advect_kernel_cu_

#include "cfdHeader.h"

extern void cudaSetBoundary ( float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim );

/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAdvect
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *den_out, float const *dens_in, float const *u_in, float const *v_in, float const *w_in
* @return   NULL
* @bref     To move the density (particles)
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAdvect ( float *den_out, float const *dens_in, float const *u_in, float const *v_in, float const *w_in )
{
	// Get index of GPU-thread
	GetIndex ( );

	int i0, j0, i1, j1;
	float s0, t0, s1, t1;

	float dt0 = DELTA_TIME * Grids_X;

	BeginSimArea ( );
	{
		float x = i - dt0 * u_in [ Index(i, j, k) ];
		float y = j - dt0 * v_in [ Index(i, j, k) ];
		if (x < 0.5f) x = 0.5f;
		if (x > SimArea_X + 0.5f) x = SimArea_X + 0.5f;

		i0 = (int)x; 
		i1 = i0+1;
		
		if ( y < 0.5f ) y=0.5f;
		if ( y > SimArea_X + 0.5f ) y = SimArea_X + 0.5f;
		
		j0 = (int)y;
		j1 = j0 + 1;
		s1 = x - i0;
		s0 = 1 - s1;
		t1 = y - j0;
		t0 = 1 - t1;

		den_out [ Index(i, j, k) ] = s0 * ( t0 * dens_in [ Index(i0, j0, k) ] +
			t1 * dens_in [ Index(i0, j1, k)] ) + s1 * ( t0 * dens_in [ Index(i1, j0, k) ] + 
			t1 * dens_in [ Index(i1, j1, k)] );
	}
	EndSimArea();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaDensAdvect
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *den_out, float const *dens_in, int boundary,
* --------- float const *u_in, float const *v_in, float const *w_in
* --------- dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Update the status of density
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaDensAdvect ( float *den_out, float const *dens_in, int boundary,
						  float const *u_in, float const *v_in, float const *w_in,
						  dim3 *gridDim, dim3 *blockDim )
{
	kernelAdvect      cudaDevice( *gridDim, *blockDim ) ( den_out, dens_in, u_in, v_in, w_in );
	cudaSetBoundary ( den_out, boundary, gridDim, blockDim );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaVelAdvect
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *den_out, float const *dens_in, int boundary,
* --------- float const *u_in, float const *v_in, float const *w_in
* --------- dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Update the status of velocity field
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaVelAdvect ( float *grid_out, float const *grid_in, int boundary,
						  float const *u_in, float const *v_in, float const *w_in,
						  dim3 *gridDim, dim3 *blockDim )
{
	kernelAdvect      cudaDevice( *gridDim, *blockDim ) ( grid_out, grid_in, u_in, v_in, w_in );
	cudaSetBoundary ( grid_out, boundary, gridDim, blockDim );
};

#endif