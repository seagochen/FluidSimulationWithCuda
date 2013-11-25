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
* <File>        DensitySolver.cpp
*/

#ifndef __Main_kernelT_cu_
#define __Main_kernelT_cu_

#include "cfdHeader.h"

using namespace std;


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelTAddSource
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    ptr_inout
* @return   NULL
* @bref     Add source to simulation grid      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelTAddSource ( float *ptr_inout )
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
* @function kernelTSetBoundary
* @author   Orlando Chen
* @date     Nov 15, 2013
* @input    grid_out, boundary
* @return   NULL
* @bref     Check and set boundary condition      
----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelTSetBoundary ( float *grid_out, int boundary )
{
	// Get index of GPU-thread
	GetIndex ( );

	// Boundary condition
	BeginSimArea();
	{
		// Slove line (gst_header, y)
		grid_out [ Index ( gst_header, j, k ) ] = boundary == 1 ? -grid_out [ Index ( sim_header, j, k ) ] : grid_out [ Index ( sim_header, j, k ) ];
		// Slove line (gst_tailer, y)
		grid_out [ Index ( gst_tailer, j, k ) ] = boundary == 1 ? -grid_out [ Index ( sim_tailer, j, k ) ] : grid_out [ Index ( sim_tailer, j, k ) ];
		// Slove line (x, gst_header)
		grid_out [ Index ( i, gst_header, k ) ] = boundary == 2 ? -grid_out [ Index ( i, sim_header, k ) ] : grid_out [ Index ( i, sim_header, k ) ];
		// Slove line (x, gst_tailer)
		grid_out [ Index ( i, gst_tailer, k ) ] = boundary == 2 ? -grid_out [ Index ( i, sim_tailer, k ) ] : grid_out [ Index ( i, sim_tailer, k ) ];
	}
	EndSimArea();

	// Slove ghost cell (gst_header, gst_header)
	grid_out [ Index ( gst_header, gst_header, k ) ] = 0.5f * ( grid_out [ Index ( sim_header, gst_header, k ) ] + grid_out [ Index ( gst_header, sim_header, k ) ] );
	// Slove ghost cell (gst_header, gst_tailer)
	grid_out [ Index ( gst_header, gst_tailer, k ) ] = 0.5f * ( grid_out [ Index ( sim_header, gst_tailer, k ) ] + grid_out [ Index ( gst_header, sim_tailer, k ) ] );
	// Slove ghost cell (gst_tailer, gst_header)
	grid_out [ Index ( gst_tailer, gst_header, k ) ] = 0.5f * ( grid_out [ Index ( sim_tailer, gst_header, k ) ] + grid_out [ Index ( gst_tailer, sim_header, k ) ] );
	// Slove ghost cell (gst_tailer, gst_tailer)
	grid_out [ Index ( gst_tailer, gst_tailer, k ) ] = 0.5f * ( grid_out [ Index ( sim_tailer, gst_tailer, k ) ] + grid_out [ Index ( gst_tailer, sim_tailer, k ) ] );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelTLineSolver
* @author   Orlando Chen
* @date     Nov 15, 2013
* @input    grid_inout, grid0_in, boudnary, a, c
* @return   NULL
* @bref     Line solver
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelTLineSolver ( float *grid_inout, float *grid0_in, int boundary, float a, float c )
{
	// Get index of GPU-thread
	GetIndex ( );

	BeginSimArea();
	{
		grid_inout [ Index ( i, j, k ) ] = ( grid0_in [ Index ( i, j, k ) ] + a * ( grid_inout [ Index ( i-1, j, k ) ] + 
			grid_inout [ Index ( i+1, j, k ) ] + grid_inout [ Index ( i, j-1, k ) ] + grid_inout [ Index ( i, j+1, k ) ] ) ) / c;	
	}
	EndSimArea();
}


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelTAdvect
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    density_out, density0_in, u_in, v_in, w_in, dt0
* @return   NULL
* @bref     Advection method      
-----------------------------------------------------------------------------------------------------------
*/

__global__ void kernelTAdvect ( float *density_out, float *density0_in, float *u_in, float *v_in, float *w_in, float dt0 )
{
	// Get index of GPU-thread
	GetIndex ( );

	int i0, j0, i1, j1;
	float s0, t0, s1, t1;

	BeginSimArea();
	{
		float x = i - dt0 * u_in[Index(i, j, k)];
		float y = j - dt0 * v_in[Index(i, j, k)];
		if (x < 0.5f) x = 0.5f;
		if (x > SimArea_X + 0.5f) x = SimArea_X+0.5f;

		i0 = (int)x; 
		i1 = i0+1;
		
		if (y < 0.5f) y=0.5f;
		if (y > SimArea_X+0.5f) y = SimArea_X+0.5f;
		
		j0 = (int)y;
		j1 = j0 + 1;
		s1 = x - i0;
		s0 = 1 - s1;
		t1 = y - j0;
		t0 = 1 - t1;

		density_out[Index(i, j, k)] = s0 * ( t0 * density0_in[Index(i0, j0, k)] +
			t1 * density0_in[Index(i0, j1, k)]) + s1 * ( t0 * density0_in[Index(i1, j0, k)] + 
			t1 * density0_in[Index(i1, j1, k)]);
	}
	EndSimArea();
};

/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAddSource
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    grid, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of kernelTAddSource      
------------------------------------------------------------------------------------------------------------
*/
void addSource ( float *grid, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernelTT on the GPU with one thread for each element.
	kernelTAddSource cudaDevice(*gridDim,  *blockDim) (grid);
};

/*
-----------------------------------------------------------------------------------------------------------
* @function cudaLineSolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    grid, grid0, boundary, a, c, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of kernelTLineSolver      
-----------------------------------------------------------------------------------------------------------
*/
void lineineSolver (float *grid, float *grid0, int boundary, float a, float c, dim3 *gridDim, dim3 *blockDim)
{
    // Launch a kernelTT on the GPU with one thread for each element.
	for (int i=0; i<20; i++)
	{
		kernelTLineSolver cudaDevice(*gridDim,  *blockDim) (grid, grid0, boundary, a, c);
	}
	kernelTSetBoundary cudaDevice(*gridDim,  *blockDim)  (grid, boundary);
}


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaDiffuse
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    grid, grid0, boundary, diff, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of diffuse method      
-----------------------------------------------------------------------------------------------------------
*/
void diffuse ( float *grid, float *grid0, int boundary, float diff, dim3 *gridDim, dim3 *blockDim )
{
	float a = DELTA_TIME * diff * SimArea_X * SimArea_X;
	lineineSolver ( grid, grid0, boundary, a, 1+4*a, gridDim, blockDim );
}

/*
------------------------------------------------------------------------------------------------------------
* @function cudaAdvect
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    density, density0, u, v, w, boundary, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of advection method      
-----------------------------------------------------------------------------------------------------------
*/
void advect ( float *density, float *density0, float *u, float *v, float *w, int boundary, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernelTT on the GPU with one thread for each element.
	float dt0 = DELTA_TIME * SimArea_X;
	kernelTAdvect      cudaDevice(*gridDim,  *blockDim) ( density, density0, u, v, w, dt0 );
	kernelTSetBoundary cudaDevice(*gridDim,  *blockDim) ( density, boundary );
}

/*
-----------------------------------------------------------------------------------------------------------
* @function DensitySolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    float *grid, float *grid0, float *u, float *v, float *w
* @return   NULL
* @bref     Calculate the advection of flow, and update the density on each cell     
-----------------------------------------------------------------------------------------------------------
*/
void DensitySolver ( float *dens, float *dens0, float *u, float *v, float *w )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_den, dens, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_den0, dens0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_u, u, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_v, v, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_w, w, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	addSource ( dev_den, &gridDim, &blockDim );               swap( dev_den, dev_den0 );
	diffuse ( dev_den, dev_den0, 0, DIFFUSION, &gridDim, &blockDim );    swap( dev_den, dev_den0 );
	advect  ( dev_den, dev_den0, dev_u, dev_v, dev_w, 0,  &gridDim, &blockDim );
    

    // cudaDeviceSynchronize waits for the kernelTT to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( dens, dev_den, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dens0, dev_den0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
	
	if ( cudaMemcpy ( u, dev_u, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( v, dev_v, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( w, dev_w, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
}




#endif