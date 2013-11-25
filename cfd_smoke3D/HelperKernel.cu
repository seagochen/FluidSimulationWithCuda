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
* <File>        HelperKernel.cu
*/

#ifndef __helper_kernel_cu_
#define __helper_kernel_cu_

#include "cfdHeader.h"

/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAnnihilation
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grid_out
* @return   NULL
* @bref     When the density is lower than the set , deemed to be zero
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAnnihilation ( float *grid_out )
{
	// Get index of GPU-thread
	GetIndex ( );

	float ANNIHILATION = 100.f / ( Grids_X * Grids_X * Grids_X );

	BeginSimArea ( );
	{
		if ( grid_out [ Index(i,j,k) ] > ANNIHILATION )
			grid_out [ Index(i,j,k) ] -= ANNIHILATION;
		else
			grid_out [ Index(i,j,k) ] = 0.f;
	}
	EndSimArea ( );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelZeroData
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grid_out
* @return   NULL
* @bref     When the density is lower than the set , deemed to be zero
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelZeroData ( float *grid_out )
{
	// Get index of GPU-thread
	GetIndex ( );
	
	grid_out [ Index(i,j,k) ] = 0.f;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAnnihilation
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grid_out, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     When the density is lower than the set , deemed to be zero
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaAnnihilation ( float *grid_out, dim3 *gridDim, dim3 *blockDim )
{
	kernelAnnihilation <<<*gridDim, *blockDim>>> ( grid_out );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaZeroData
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grid_out, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     When the density is lower than the set , deemed to be zero
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaZeroData ( float *grid_out, dim3 *gridDim, dim3 *blockDim )
{
	kernelZeroData <<<*gridDim, *blockDim>>> ( grid_out );
};

#endif