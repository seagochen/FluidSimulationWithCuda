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
* <Last>		Nov 19, 2013
* <File>        cfdMainKernel.cu
*/

#ifndef __cfd_Main_Kernel_cu_
#define __cfd_Main_Kernel_cu_

#include "cfdHeader.h"

using namespace std;

/*
  -----------------------------------------------------------------------------------------------------------
   Define something
  -----------------------------------------------------------------------------------------------------------
*/

#define is       ==            /* equal to */
#define like     ==            /* equal to */
#define gte      >=            /* greater than or equal to  */
#define gt       >             /* greater than */
#define lse      <=            /* less than or equal to */
#define ls       <             /* less than */
#define and      &&            /* logical and */
#define or       ||            /* logical or */

#define gst0   0               /* ghost cell, No. #0 */
#define gstl   Grids_X - 1     /* ghost cell, No. #last */
#define rsc0   1               /* simulation cell, No. #0 */
#define rscl   SimArea_X       /* simulation cell, No. #last */


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAddSource
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    ptr_inout
* @return   NULL
* @bref     Add source to simulation grid      
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
				ptr_inout [ Index (i, j, k) ] = SOURCE * DELTA_TIME * 0.1f;
		}
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelSetBoundary
* @author   Orlando Chen
* @date     Nov 15, 2013
* @input    grid_out, boundary
* @return   NULL
* @bref     Check and set boundary condition      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelSetBoundary ( float *grid_out, int boundary )
{
	// Get index of GPU-thread
	GetIndex ( );

	// Boundary condition
	if ( i gte 1 && i lse SimArea_X && j gte 1 && j lse SimArea_X )
	{
		// Slove line (gst0, y)
		grid_out [ Index ( gst0, j, k ) ] = boundary is 1 ? -grid_out [ Index ( rsc0, j, k ) ] : grid_out [ Index ( rsc0, j, k ) ];
		// Slove line (gstl, y)
		grid_out [ Index ( gstl, j, k ) ] = boundary is 1 ? -grid_out [ Index ( rscl, j, k ) ] : grid_out [ Index ( rscl, j, k ) ];
		// Slove line (x, gst0)
		grid_out [ Index ( i, gst0, k ) ] = boundary is 2 ? -grid_out [ Index ( i, rsc0, k ) ] : grid_out [ Index ( i, rsc0, k ) ];
		// Slove line (x, gstl)
		grid_out [ Index ( i, gstl, k ) ] = boundary is 2 ? -grid_out [ Index ( i, rscl, k ) ] : grid_out [ Index ( i, rscl, k ) ];
	}
	// Slove ghost cell (gst0, gst0)
	grid_out [ Index ( gst0, gst0, k ) ] = 0.5f * ( grid_out [ Index ( rsc0, gst0, k ) ] + grid_out [ Index ( gst0, rsc0, k ) ] );
	// Slove ghost cell (gst0, gstl)
	grid_out [ Index ( gst0, gstl, k ) ] = 0.5f * ( grid_out [ Index ( rsc0, gstl, k ) ] + grid_out [ Index ( gst0, rscl, k ) ] );
	// Slove ghost cell (gstl, gst0)
	grid_out [ Index ( gstl, gst0, k ) ] = 0.5f * ( grid_out [ Index ( rscl, gst0, k ) ] + grid_out [ Index ( gstl, rsc0, k ) ] );
	// Slove ghost cell (gstl, gstl)
	grid_out [ Index ( gstl, gstl, k ) ] = 0.5f * ( grid_out [ Index ( rscl, gstl, k ) ] + grid_out [ Index ( gstl, rscl, k ) ] );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelLineSolver
* @author   Orlando Chen
* @date     Nov 15, 2013
* @input    grid_inout, grid0_in, boudnary, a, c
* @return   NULL
* @bref     Line solver
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelLineSolver ( float *grid_inout, float *grid0_in, int boundary, float a, float c )
{
	// Get index of GPU-thread
	GetIndex ( );

	if ( i gte 1 && i lse SimArea_X && j gte 1 && j lse SimArea_X )
	{
		grid_inout [ Index ( i, j, k ) ] = ( grid0_in [ Index ( i, j, k ) ] + a * ( grid_inout [ Index ( i-1, j, k ) ] + 
			grid_inout [ Index ( i+1, j, k ) ] + grid_inout [ Index ( i, j-1, k ) ] + grid_inout [ Index ( i, j+1, k ) ] ) ) / c;	
	}
}


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAdvect
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    density_out, density0_in, u_in, v_in, w_in, dt0
* @return   NULL
* @bref     Advection method      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAdvect ( float *density_out, float *density0_in, float *u_in, float *v_in, float *w_in, float dt0 )
{
	// Get index of GPU-thread
	GetIndex ( );

	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1;

	if ( i gte 1 && i lse SimArea_X && j gte 1 && j lse SimArea_X )
	{
		x = i - dt0 * u_in[Index(i, j, k)];
		y = j - dt0 * v_in[Index(i, j, k)];
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
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelProjectPt1
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    u, v, w, u0, v0, w0
* @return   NULL
* @bref     CFD projection part I      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelProjectPt1 ( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Get index of GPU-thread
	GetIndex ( );
	
	if ( i gte 1 && i lse SimArea_X && j gte 1 && j lse SimArea_X )
	{
		v0 [ Index (i, j, k) ] = -0.5f * ( u [ Index ( i+1, j, k ) ] - u [ Index ( i-1, j, k ) ] + v [ Index ( i, j+1, k ) ] 
		- v [ Index ( i, j-1, k ) ] ) / SimArea_X;
		u0 [ Index (i, j, k) ] = 0;
	}
}

/*
-----------------------------------------------------------------------------------------------------------
* @function kernelProjectPt2
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    u, v, w, u0, v0, w0
* @return   NULL
* @bref     CFD projection part II      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelProjectPt2( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Get index of GPU-thread
	GetIndex ( );
	
	if ( i gte 1 && i lse SimArea_X && j gte 1 && j lse SimArea_X )
	{
			u [ Index ( i, j, k ) ] -= 0.5f * SimArea_X * ( u0 [ Index ( i+1, j, k ) ] - u0 [ Index ( i-1, j, k ) ] );
			v [ Index ( i, j, k ) ] -= 0.5f * SimArea_X * ( u0 [ Index ( i, j+1, k ) ] - u0 [ Index ( i, j-1, k ) ] );
	}
}


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAddSource
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    grid, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of kernelAddSource      
-----------------------------------------------------------------------------------------------------------
*/
void cudaAddSource ( float *grid, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	kernelAddSource cudaDevice(*gridDim,  *blockDim) (grid);
};

/*
-----------------------------------------------------------------------------------------------------------
* @function cudaLineSolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    grid, grid0, boundary, a, c, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of kernelLineSolver      
-----------------------------------------------------------------------------------------------------------
*/
void cudaLineSolver (float *grid, float *grid0, int boundary, float a, float c, dim3 *gridDim, dim3 *blockDim)
{
    // Launch a kernel on the GPU with one thread for each element.
	for (int i=0; i<20; i++)
	{
		kernelLineSolver cudaDevice(*gridDim,  *blockDim) (grid, grid0, boundary, a, c);
	}
	kernelSetBoundary cudaDevice(*gridDim,  *blockDim)  (grid, boundary);
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
void cudaDiffuse ( float *grid, float *grid0, int boundary, float diff, dim3 *gridDim, dim3 *blockDim )
{
	float a = DELTA_TIME * diff * SimArea_X * SimArea_X;
	cudaLineSolver ( grid, grid0, boundary, a, 1+4*a, gridDim, blockDim );
}

/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAdvect
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    density, density0, u, v, w, boundary, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of advection method      
-----------------------------------------------------------------------------------------------------------
*/
void cudaAdvect ( float *density, float *density0, float *u, float *v, float *w, int boundary, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	float dt0 = DELTA_TIME*SimArea_X;
	kernelAdvect      cudaDevice(*gridDim,  *blockDim) ( density, density0, u, v, w, dt0 );
	kernelSetBoundary cudaDevice(*gridDim,  *blockDim) ( density, boundary );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaProject
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    u, v, w, u0, v0, w0, gridDim, blockDim
* @return   NULL
* @bref     C++ encapsulation of projection method
*      
-----------------------------------------------------------------------------------------------------------
*/
void cudaProject ( float *u, float *v, float *w, float *u0, float *v0, float *w0, dim3 *gridDim, dim3 *blockDim )
{
	kernelProjectPt1   cudaDevice(*gridDim,  *blockDim) ( u, v, w, u0, v0, w0 );
	kernelSetBoundary  cudaDevice(*gridDim,  *blockDim) ( v0, 0 );
	kernelSetBoundary  cudaDevice(*gridDim,  *blockDim) ( u0, 0 );
	kernelLineSolver   cudaDevice(*gridDim,  *blockDim) ( u0, v0, 0, 1, 4 );
	kernelProjectPt2   cudaDevice(*gridDim,  *blockDim) ( u, v, w, u0, v0, w0 );
	kernelSetBoundary  cudaDevice(*gridDim,  *blockDim) ( u, 1 );
	kernelSetBoundary  cudaDevice(*gridDim,  *blockDim) ( v, 2 );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function DensitySolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    grid, grid0, u, v, w
* @return   NULL
* @bref     Calculate the advection of flow, and update the density on each cell     
-----------------------------------------------------------------------------------------------------------
*/
void DensitySolver ( float *grid, float *grid0, float *u, float *v, float *w )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_grid, grid, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_grid0, grid0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_u, u, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_v, v, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_w, w, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );


	cudaAddSource ( dev_grid, &gridDim, &blockDim );
	swap ( dev_grid0, dev_grid ); cudaDiffuse ( dev_grid, dev_grid0, 0, DIFFUSION, &gridDim, &blockDim );
	swap ( dev_grid0, dev_grid ); cudaAdvect  ( dev_grid, dev_grid0, dev_u, dev_v, dev_w, 0, &gridDim, &blockDim );
    

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( grid, dev_grid, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( grid0, dev_grid0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
	
	if ( cudaMemcpy ( u, dev_u, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( v, dev_v, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( w, dev_w, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function VelocitySolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    u, v, w, u0, v0, w0
* @return   NULL
* @bref     Calculate the advection of flow, and update the velocity on each cell
*      
-----------------------------------------------------------------------------------------------------------
*/
void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_u0, u0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_v0, v0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_w0, w0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_u, u, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_v, v, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_w, w, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );


	cudaAddSource ( dev_u, &gridDim, &blockDim ); cudaAddSource ( dev_v, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); cudaDiffuse ( dev_u, dev_u0, 1, VISCOSITY, &gridDim, &blockDim );
	swap ( dev_v0, dev_v ); cudaDiffuse ( dev_v, dev_v0, 2, VISCOSITY, &gridDim, &blockDim );
	cudaProject ( dev_u, dev_v, dev_w, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); swap ( dev_v0, dev_v );
	cudaAdvect ( dev_u, dev_u0, dev_u0, dev_v0, dev_w0, 1, &gridDim, &blockDim );
	cudaAdvect ( dev_v, dev_v0, dev_u0, dev_v0, dev_w0, 2, &gridDim, &blockDim );
	cudaProject ( dev_u, dev_v, dev_w, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );


    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( u0, dev_u0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( v0, dev_v0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( w0, dev_w0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
	
	if ( cudaMemcpy ( u, dev_u, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( v, dev_v, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( w, dev_w, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
}


/*
  -----------------------------------------------------------------------------------------------------------
   Undef the definitions
  -----------------------------------------------------------------------------------------------------------
*/

#undef gst0   /* ghost cell, No. #0 */
#undef gstl   /* ghost cell, No. #last */
#undef rsc0   /* simulation cell, No. #0 */
#undef rscl   /* simulation cell, No. #last */

#undef is     /* equal to */
#undef like   /* equal to */
#undef gte    /* greater than or equal to  */
#undef gt     /* greater than */
#undef lse    /* less than or equal to */
#undef ls     /* less than */
#undef and    /* logical and */
#undef or     /* logical or */

#endif