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
* <File>        ProjectFieldKernel.cu
*/

#ifndef __project_field_kernel_cu_
#define __project_field_kernel_cu_

#include "cfdHeader.h"

extern void cudaSetBoundary ( float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim );

/*
-----------------------------------------------------------------------------------------------------------
* @function kernelDivergence
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grad_out, float *proj_out, float const *u_in, float const *v_in, float const *w_in
* @return   NULL
* @bref     To calculate the gradient of velocity U
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelDivergence ( float *grad_out, float *proj_out, float const *u_in, float const *v_in, float const *w_in )
{
	// Get the thread ID
	GetIndex ( );

	float h = 1.0 / Grids_X;

	BeginSimArea ();
	{
		grad_out [ Index(i,j,k) ] = -0.5 * h * ( 
			u_in [ Index(i+1,j,k) ] - u_in [ Index(i-1,j,k) ] +  // gradient of u
			v_in [ Index(i,j+1,k) ] - v_in [ Index(i,j-1,k) ] ); // gradient of v

		proj_out [ Index(i,j,k) ] = 0.f;
	}
	EndSimArea ();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelConservField
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float const *grad_in, float *proj_out, float const *u_in, float const *v_in, float const *w_in
* @return   NULL
* @bref     To calculate the mass conserving field
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelConservField ( float const *grad_in, float *proj_out, float const *u_in, float const *v_in, float const *w_in )
{
	// Get the thread ID
	GetIndex ( );

	float h = 1.0 / Grids_X;

	BeginSimArea ();
	{
		proj_out [ Index(i,j,k) ] = ( grad_in [ Index(i,j,k) ] + 
			proj_out [ Index(i-1,j,k) ] +
			proj_out [ Index(i+1,j,k) ] +
			proj_out [ Index(i,j-1,k) ] +
			proj_out [ Index(i,j+1,k) ] ) / 4.f;
	}
	EndSimArea ();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelProjectVelocity
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float const *grad_in, float const *proj_out, float *u_in, float *v_in, float *w_in
* @return   NULL
* @bref     To calculate the mass conserving field
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelProjectVelocity ( float const *grad_in, float const *proj_in, float *u_out, float *v_out, float *w_out )
{
	// Get the thread ID
	GetIndex ( );

	float h = 1.0 / Grids_X;

	BeginSimArea ();
	{
		u_out [ Index(i,j,k) ] -= 0.5f * ( proj_in [ Index(i+1, j, k) ] - proj_in [ Index(i-1, j, k) ] ) / h;
		v_out [ Index(i,j,k) ] -= 0.5f * ( proj_in [ Index(i, j+1, k) ] - proj_in [ Index(i, j-1, k) ] ) / h;
	}
	EndSimArea ();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaProjectField
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float const *grad_in, float *proj_out, float const *u_in, float const *v_in, float const *w_in
* @return   NULL
* @bref     To calculate the mass conserving field
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaProjectField 
	( float *grad_in, float *proj_out, float *u_in, float *v_in, float *w_in, dim3 *gridDim, dim3 *blockDim )
{
	kernelDivergence cudaDevice(*gridDim, *blockDim) (grad_in, proj_out, u_in, v_in, w_in);

	cudaSetBoundary ( grad_in, 0, gridDim, blockDim );
	cudaSetBoundary ( proj_out, 0, gridDim, blockDim );

	for ( int i = 0; i < 20; i++ )
	{
		kernelConservField cudaDevice(*gridDim, *blockDim) (grad_in, proj_out, u_in, v_in, w_in);
		cudaSetBoundary ( proj_out, 0, gridDim, blockDim );
	}

	kernelProjectVelocity cudaDevice(*gridDim, *blockDim) (grad_in, proj_out, u_in, v_in, w_in);
	cudaSetBoundary ( u_in, 1, gridDim, blockDim );
	cudaSetBoundary ( v_in, 2, gridDim, blockDim );
};


#endif