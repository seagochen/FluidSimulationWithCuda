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
* @function kernelAddSource
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    float *ptr_inout
* @return   NULL
* @bref     Add new density of flow as well as initial velocity, need to call this function first of all 
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAddSource ( float *ptr_inout )
{
	// Get index of GPU-thread
	GetIndex ( );

	// Coordinates arround the (64, 64, 64), r is 5
	if ( i > 54 and i < 74 ) if ( k > 54 and k < 74 )
	{
		int x = i - 64;
		int y = k - 64;
		float r = sqrtf ( x * x + y * y );
		
		// Within the correct distance
		if ( r >= 0 && r <= 5  )
		{
			// Add source from layer 0 - 4
			if ( j < 5 )
				ptr_inout [ Index (i, j, k) ] += SOURCE * DELTA_TIME;
		}
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAddSource
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *ptr_inout, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Encapsulation the CUDA routine (addsource)
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaAddSource ( float *ptr_inout, dim3 *gridDim, dim3 *blockDim )
{
	kernelAddSource cudaDevice(*gridDim, *blockDim) (ptr_inout);
};

#endif