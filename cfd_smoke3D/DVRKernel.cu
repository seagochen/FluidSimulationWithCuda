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

__global__ void kernelZeroBuffer ( float *buffer_inout )
{
	GetIndex ( );

	buffer_inout [ cudaIndex3D (i, j, 0, Grids_X) ] = 0.f;
};


__global__ void kernelDensityInterpolate ( float *den3D_in, float *den2D_out )
{
	GetIndex ( );
	
	den2D_out [ cudaIndex3D (i, j, 0, Grids_X) ] += den3D_in [ cudaIndex3D (i, j, k, Grids_X) ];
};


void DensityInterpolate ( void )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	// Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_grid, host_den, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed!" );

	printf ("passed 1\n");

	// Launch kernels
	kernelZeroBuffer cudaDevice(gridDim, blockDim) (dev_display_temp2D1);
	kernelDensityInterpolate cudaDevice(gridDim, blockDim) (dev_grid, dev_display_temp2D1);

	printf ("passed 2\n");

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

	printf ("passed 3\n");

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( host_display_den, dev_display_temp2D1, DIS_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	printf ("passed 4\n");
};


__global__ void kernelVelocityInterpolate ( float *u3D_in, float *v3D_in, float *u2D_out, float *v2D_out )
{
	GetIndex ( );

	u2D_out [ cudaIndex3D (i, j, 0, Grids_X) ] = u3D_in [ cudaIndex3D (i, j, 10, Grids_X) ];
	v2D_out [ cudaIndex3D (i, j, 0, Grids_X) ] = v3D_in [ cudaIndex3D (i, j, 10, Grids_X) ];
};


void VelocityInterpolate ( void )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	// Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_u, host_u, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed!" );
	if ( cudaMemcpy ( dev_v, host_v, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed!" );

	// Launch kernels
	kernelVelocityInterpolate cudaDevice(gridDim, blockDim) (dev_u, dev_v, dev_display_temp2D1, dev_display_temp2D2);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( host_display_u, dev_display_temp2D1, DIS_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
	if ( cudaMemcpy ( host_display_v, dev_display_temp2D2, DIS_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
};


#define Index(i,j) cudaIndex3D(i,j,0,Grids_X)


void DrawVelocity ( void )
{
	VelocityInterpolate ( );

	float x, y, h;

	h = 1.0f / SimArea_X;

	glColor3f ( 0.0f, 0.0f, 1.0f );
	glLineWidth ( 1.0f );

	glBegin ( GL_LINES );
	{
		for ( int i = 1 ; i <= SimArea_X ; i++ )
		{
			x = ( i - 0.5f ) * h;
			for ( int j = 1 ; j <= SimArea_X ; j++ )
			{
				y = (j-0.5f)*h;
				glVertex2f ( x, y );
				glVertex2f ( x + host_display_u [ Index ( i, j ) ], y + host_display_v [ Index ( i, j ) ] );
			}
		}
	}
	glEnd ( );
}


void DrawDensity ( void )
{
	DensityInterpolate ( );

	float x, y, h, d00, d01, d10, d11;

	h = 1.0f / SimArea_X;

	glBegin ( GL_QUADS );
	{
		for ( int i=0 ; i<=SimArea_X ; i++ )
		{
			x = (i-0.5f)*h;
			for ( int j=0 ; j<=SimArea_X ; j++ )
			{
				y = (j-0.5f)*h;
				d00 = host_display_den [ Index ( i, j ) ];
				d01 = host_display_den [ Index ( i, j+1 ) ];
				d10 = host_display_den [ Index ( i+1, j ) ];
				d11 = host_display_den [ Index ( i+1, j+1 ) ];

				glColor3f(d00, d00, d00); glVertex2f(x, y);
				glColor3f(d10, d10, d10); glVertex2f(x+h, y);
				glColor3f(d11, d11, d11); glVertex2f(x+h, y+h);
				glColor3f(d01, d01, d01); glVertex2f(x, y+h);
			}
		}
	}
	glEnd();
}


#endif