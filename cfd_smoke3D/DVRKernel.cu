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
* <First>       Nov 15, 2013
* <Last>		Nov 15, 2013
* <File>        DVRKernel.cu
*/

#ifndef __rendering_Kernel_cu_
#define __rendering_Kernel_cu_

#include "cfdHeaders.h"
#include "cudaHelper.h"
#include "macroDef.h"

#define is       ==            /* equal to */
#define like     ==            /* equal to */
#define gte      >=            /* greater than or equal to  */
#define gt       >             /* greater than */
#define lse      <=            /* less than or equal to */
#define ls       <             /* less than */
#define and      &&            /* logical and */
#define or       ||            /* logical or */


__global__ void kernelZeroBuffer ( float *buffer_inout )
{
	GetIndex ( );

	buffer_inout [ cudaIndex3D (i, j, k, Grids_X) ] = 0.f;
};


__global__ void kernelDensityInterpolate ( float *den3D_in, float *den2D_out )
{
	GetIndex ( );
	
	den2D_out [ cudaIndex3D (i, j, 0, Grids_X) ] += den3D_in [ cudaIndex3D (i, j, k, Grids_X) ];
};


__global__ void kernelVelocityInterpolate ( float *u3D_in, float *v3D_in, float *u2D_out, float *v2D_out )
{
	GetIndex ( );

	u2D_out [ cudaIndex2D (i, j, Grids_X) ] += u3D_in [ cudaIndex3D (i, j, k, Grids_X) ];
	v2D_out [ cudaIndex2D (i, j, Grids_X) ] += v3D_in [ cudaIndex3D (i, j, k, Grids_X) ];
};


__global__ void kernelTester ( float *buffer_inout )
{
	GetIndex ( );

	buffer_inout [ cudaIndex3D (i, j, k, Grids_X) ] = cudaIndex3D (i, j, k, Grids_X) / 50000.f;
};



void DensityInterpolate ( void )
{
	extern void FreeResources  ( void );

	// Define the computing unit size
	cudaDeviceDim2D ( );

	float *devPtr = 0;

	if ( cudaMalloc ( ( void ** ) &devPtr, sizeof(float) * DIS_SIZE ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMalloc was failed!" );

    // Copy input vectors from host memory to GPU buffers.


	kernelTester cudaDevice(gridDim, blockDim) (devPtr);
    

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( host_disD, devPtr, sizeof(float) * DIS_SIZE, cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
/*

	FILE *stream = fopen ( "test.txt", "w" ); 

	for ( int i = 0; i < Grids_X; i++ ) 
	{
		for ( int j = 0; j < Grids_X; j++ )
		{
			fprintf ( stream, "%f ", host_disD [ cudaIndex3D ( i, j, 0, Grids_X) ] );
		}

		fprintf ( stream, "\n" );
	}

	fclose ( stream );
	cudaFree ( devPtr );
	FreeResources ( );
	exit (0);
	*/
};


void VelocityInterpolate ( void )
{
	for ( int i = 0; i < Grids_X; i++ )
	{
		for ( int j = 0; j < Grids_X; j++ )
		{
			float var0 = 0.f, var1 = 0.f;
			
			for ( int k = 0; k < Grids_X; k++ )
			{
				var0 = host_u [ cudaIndex3D (i, j, k, Grids_X) ];
				var1 = host_v [ cudaIndex3D (i, j, k, Grids_X) ];
			}

			host_disu [ cudaIndex2D (i, j, Grids_X)] = var0;
			host_disv [ cudaIndex2D (i, j, Grids_X)] = var1;
		}
	}
};


#undef is     /* equal to */
#undef like   /* equal to */
#undef gte    /* greater than or equal to  */
#undef gt     /* greater than */
#undef lse    /* less than or equal to */
#undef ls     /* less than */
#undef and    /* logical and */
#undef or     /* logical or */

#undef GetIndex()

#endif