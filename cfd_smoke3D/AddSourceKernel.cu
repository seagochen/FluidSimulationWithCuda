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
* <File>        AddSourceKernel.cu
*/

#ifndef __add_source_kernel_cu_
#define __add_source_kernel_cu_

#include "cfdHeader.h"


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAddDensity
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    float *den_out
* @return   NULL
* @bref     Add new density of flow as well as initial velocity, need to call this function first of all 
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAddDensity ( float *den_out )
{
	// Get index of GPU-thread
	GetIndex ( );

	int half = Grids_X / 2;

	// Coordinates arround the center, r is 5
	if ( i > half - 10 and i < half + 10 ) if ( k > half - 10 and k < half + 10 )
	{
		int x = i - half;
		int y = k - half;
		float r = sqrtf ( x * x + y * y );
		
		// Within the correct distance
		if ( r >= 0 && r <= 5  )
		{
			// Add source from layer 0 - 4
			if ( j < 4 )
				den_out [ Index (i, j, k) ] = SOURCE * DELTA_TIME;
		}
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAddVelocity
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    float *u_out, float *v_out, float *w_out
* @return   NULL
* @bref     Add new density of flow as well as initial velocity, need to call this function first of all 
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAddVelocity ( float *u_out, float *v_out, float *w_out )
{
	// Get index of GPU-thread
	GetIndex ( );

	int half = Grids_X / 2;

	// Coordinates arround the center, r is 2
	if ( i > half - 10 and i < half + 10 ) if ( k > half - 10 and k < half + 10 )
	{
		int x = i - half;
		int y = k - half;
		float r = sqrtf ( x * x + y * y );
		
		// Within the correct distance
		if ( r >= 0 && r <= 2  )
		{
			// Add source from layer 0 - 4
			if ( j < 2 )
			{
				v_out [ Index (i, j, k) ] = SOURCE * DELTA_TIME;

				if ( i > half - 10 and i <= half )
				{
					u_out [ Index (i, j, k) ] = - SOURCE * DELTA_TIME * 0.1f;
					w_out [ Index (i, j, k) ] = - SOURCE * DELTA_TIME * 0.1f;
				}
				else
				{
					u_out [ Index (i, j, k) ] = SOURCE * DELTA_TIME * 0.1f;
					w_out [ Index (i, j, k) ] = SOURCE * DELTA_TIME * 0.1f;
				}
			}
		}
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAddSource
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *den_out, float *u_out, float *v_out, float *w_out, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Add new velocity and density to the field
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaAddSource ( float *den_out, float *u_out, float *v_out, float *w_out, 
	dim3 *gridDim, dim3 *blockDim )
{
	if ( den_out != NULL )
		kernelAddDensity cudaDevice(*gridDim, *blockDim) (den_out);
	if ( u_out != NULL and v_out != NULL and w_out != NULL )
		kernelAddVelocity cudaDevice(*gridDim, *blockDim) (u_out, v_out, w_out);
};

#endif