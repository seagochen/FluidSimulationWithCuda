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
* <Last>		Dec 20, 2013
* <File>        kernel.cu
*/

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cfd.h"
#include "cudabox.h"
#include "funcdef.h"


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



/*
 -----------------------------------------------------------------------------------------------------------
* @function kernelSetBoundary
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grid_out, int boundary
* @return   NULL
* @bref     Check and set boundary condition      
------------------------------------------------------------------------------------------------------------
*/
__global__ void kernelSetBoundary ( float *grid_out, int const boundary )
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
* @function cudaSetBoundary
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Boundary condition
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaSetBoundary ( float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim )
{
	kernelSetBoundary cudaDevice(*gridDim, *blockDim) (grid_out, boundary);
};


/*
-----------------------------------------------------------------------------------------------------------
* @function subInterpolation
* @author   Orlando Chen
* @date     Dec 2, 2013
* @input    float v0, float v1, float w0, float w1
* @return   float
* @bref     對樣本做線性插值，需要輸入樣本值v0, v1，以及插值權重w0, w1
-----------------------------------------------------------------------------------------------------------
*/
__device__ float subInterpolation ( float v0, float v1, float w0, float w1 )
{
	return v0 * w0 + v1 * w1;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAdvect
* @author   Orlando Chen
* @date     Dec 2, 2013
* @input    float *den_out, float const *dens_in, float const *u_in, float const *v_in, float const *w_in
* @return   NULL
* @bref     To move the density (particles)
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAdvect ( float *den_out, float const *dens_in, float const *u_in, float const *v_in, float const *w_in )
{
	// Get index of GPU-thread
	GetIndex ( );
	
	float dt0 = DELTA_TIME * Grids_X;

	BeginSimArea ( );
	{
		// <latex>{P}' = P_o - \bigtriangleup h \cdot \vec{U}</latex>, 計算單位時間內P點移動的位置
		float x = i - dt0 * u_in [ Index ( i, j, k ) ];
		float y = j - dt0 * v_in [ Index ( i, j, k ) ];
		float z = k - dt0 * w_in [ Index ( i, j, k ) ];

		// 考慮到系統是封閉區域，所以需要做邊界檢測
		if ( x < 0.5f ) x = 0.5f;
		if ( y < 0.5f ) y = 0.5f;
		if ( z < 0.5f ) z = 0.5f;
		if ( x > SimArea_X + 0.5f ) x = SimArea_X + 0.5f;		
		if ( y > SimArea_X + 0.5f ) y = SimArea_X + 0.5f;
		if ( z > SimArea_X + 0.5f ) z = SimArea_X + 0.5f;

		// 新位置<latex>{P}'</latex>的附加格點位置
		int i0 = (int)x; 
		int j0 = (int)y;
		int k0 = (int)z;
		int i1 = i0 + 1;
		int j1 = j0 + 1;
		int k1 = k0 + 1;
		
		// 計算插值所需的權重
		float u1 = x - i0;
		float u0 = 1 - u1;
		float v1 = y - j0;
		float v0 = 1 - v1;
		float w1 = z - k0;
		float w0 = 1 - w1;

		// 對點<latex>{P}'</latex>，w方向做插值計算
		float tempi0j0 = subInterpolation ( dens_in [ Index (i0, j0, k0) ], dens_in [ Index (i0, j0, k1) ], w0, w1 );
		float tempi0j1 = subInterpolation ( dens_in [ Index (i0, j1, k0) ], dens_in [ Index (i0, j1, k1) ], w0, w1 );
		float tempi1j0 = subInterpolation ( dens_in [ Index (i1, j0, k0) ], dens_in [ Index (i1, j0, k1) ], w0, w1 );
		float tempi1j1 = subInterpolation ( dens_in [ Index (i1, j1, k0) ], dens_in [ Index (i1, j1, k1) ], w0, w1 );

		// 對點<latex>{P}'</latex>，v方向做插值計算
		float tempi0   = subInterpolation ( tempi0j0, tempi0j1, v0, v1 );
		float tempi1   = subInterpolation ( tempi1j0, tempi1j1, v0, v1 );

		// 對點<latex>{P}'</latex>，u方向做插值計算, 並獲得最終結果
		den_out [ Index(i, j, k) ] = subInterpolation ( tempi0, tempi1, u0, u1 );
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


/*
-----------------------------------------------------------------------------------------------------------
* @function subDivergence
* @author   Orlando Chen
* @date     Dec 2, 2013
* @input    float *grid_in, int i, int j, int k
* @return   float
* @bref     <latex> \nabla {P}'|_{\left(i,j,k \right)} </latex>
-----------------------------------------------------------------------------------------------------------
*/
__device__ float subDivergence ( float *grid_in, int i, int j, int k )
{
	return
		grid_in[Index(i-1, j, k)] + grid_in[Index(i+1, j, k)] + 
		grid_in[Index(i, j-1, k)] + grid_in[Index(i, j+1, k)] +
		grid_in[Index(i, j, k-1)] + grid_in[Index(i, j, k+1)];
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelDiffuse
* @author   Orlando Chen
* @date     Nov 25, 2013
* @input    float *grid_out, float const *grid_in
* @return   NULL
* @bref     To diffuse (smooth) the simulation result
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelLineSolver ( float *grid_out, float const *grid_in, float const ratio, float const div )
{
	// Get index of GPU-thread
	GetIndex ( );

	BeginSimArea ( );
	{
		grid_out [ Index(i, j, k) ] = ( grid_in [ Index(i, j, k) ] + ratio * ( subDivergence (grid_out, i, j, k) )) / div;
	}
	EndSimArea ( );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaViscosity
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Viscosity
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaViscosity ( float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim )
{
	float ratio = DELTA_TIME * VISCOSITY * SimArea_X * SimArea_X;
	float div   = 1.f + 6.f * ratio;

	for ( int i = 0; i < 20; i++ )
	{
		kernelLineSolver cudaDevice(*gridDim, *blockDim) ( grid_out, grid_in, ratio, div );
		cudaSetBoundary  ( grid_out, boundary, gridDim, blockDim );
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaDiffuse
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Diffusion
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaDiffuse ( float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim )
{
	float ratio = DELTA_TIME * DIFFUSION * SimArea_X * SimArea_X;
	float div   = 1.f + 6.f * ratio;

//	for ( int i = 0; i < 20; i++ )
	{
		kernelLineSolver cudaDevice(*gridDim, *blockDim) ( grid_out, grid_in, ratio, div );
		cudaSetBoundary  ( grid_out, boundary, gridDim, blockDim );
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelDivergence
* @author   Orlando Chen
* @date     Dec 2, 2013
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

	BeginSimArea ( );
	{
		grad_out [ Index( i, j, k ) ] = -0.5 * h * ( 
			u_in [ Index( i+1, j, k ) ] - u_in [ Index( i-1, j, k ) ] + // gradient of u
			v_in [ Index( i, j+1, k ) ] - v_in [ Index( i, j-1, k ) ] + // gradient of v
			w_in [ Index( i, j, k+1 ) ] - w_in [ Index( i, j, k-1 ) ]); // gradient of w

		proj_out[ Index( i, j, k ) ] = 0.f;
	}
	EndSimArea ( );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelConservField
* @author   Orlando Chen
* @date     Dec 2, 2013
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

	BeginSimArea ( );
	{
		proj_out [ Index ( i, j, k ) ] = ( grad_in [ Index ( i, j, k ) ] + 
			proj_out [ Index ( i-1, j, k ) ] + proj_out [ Index ( i+1, j, k ) ] +
			proj_out [ Index ( i, j-1, k ) ] + proj_out [ Index ( i, j+1, k ) ] +
			proj_out [ Index ( i, j, k-1 ) ] + proj_out [ Index ( i, j, k+1 ) ]) / 6.f;
	}
	EndSimArea ( );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelProjectVelocity
* @author   Orlando Chen
* @date     Dec 2, 2013
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

	BeginSimArea ( );
	{
		u_out [ Index ( i, j, k ) ] -= 0.5f * ( proj_in [ Index ( i+1, j, k ) ] - proj_in [ Index ( i-1, j, k ) ] ) / h;
		v_out [ Index ( i, j, k ) ] -= 0.5f * ( proj_in [ Index ( i, j+1, k ) ] - proj_in [ Index ( i, j-1, k ) ] ) / h;
		w_out [ Index ( i, j, k ) ] -= 0.5f * ( proj_in [ Index ( i, j, k+1 ) ] - proj_in [ Index ( i, j, k-1 ) ] ) / h;
	}
	EndSimArea ( );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaProjectField
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grad_in, float *proj_out, float *u_in, float *v_in, float *w_in,
* --------- dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Update velocity field
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaProjectField ( float *grad_in, float *proj_out, float *u_in, float *v_in, float *w_in, dim3 *gridDim, dim3 *blockDim )
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


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelInterpolation
* @author   Orlando Chen
* @date     Dec 18, 2013
* @input    float *dens, unsigned char *data
* @return   NULL
* @bref
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelInterpolation ( float *dens, unsigned char *data )
{
	GetIndex();

	int temp = (int) ceil ( dens [ Index (i, j, k) ] );
	data [ Index (i, j, k) ] = (unsigned char) temp;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function DensitySolver
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *dens, float *dens0, float *u, float *v, float *w
* @return   NULL
* @bref     Add Some particles in the velocity field and calculate how it effects on these particles
-----------------------------------------------------------------------------------------------------------
*/
void DensitySolver ( float *dens, float *dens0, float *u, float *v, float *w )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
	// Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_den, dens, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );
	
	if ( cudaMemcpy ( dev_den0, dens0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_u, u, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_v, v, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_w, w, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
	
//	cudaAddSource ( dev_den, NULL, NULL, NULL, &gridDim, &blockDim );
//	swap( dev_den, dev_den0 );
//	cudaDiffuse ( dev_den, dev_den0, 0, &gridDim, &blockDim ); swap( dev_den, dev_den0 );
//  cudaDensAdvect (dev_den, dev_den0, 0, dev_u, dev_v, dev_w, &gridDim, &blockDim );

	// cudaDeviceSynchronize waits for the kernelTT to finish, and returns
	// any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );
	
	// Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( dens, dev_den, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );
	if ( cudaMemcpy ( dens0, dev_den0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );
//	
//	if ( cudaMemcpy ( u, dev_u, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( v, dev_v, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( w, dev_w, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function VelocitySolver
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *u, float *v, float *w, float *u0, float *v0, float *w0
* @return   NULL
* @bref     To solve the velocity field of fluid
-----------------------------------------------------------------------------------------------------------
*/
void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
//	if ( cudaMemcpy ( dev_u0, u0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_v0, v0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_w0, w0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_u, u, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_v, v, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( dev_w, w, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
/*
	cudaAddSource (NULL, dev_u, dev_v, dev_w, &gridDim, &blockDim );  
	swap ( dev_u0, dev_u ); cudaViscosity ( dev_u, dev_u0, 1, &gridDim, &blockDim );
	swap ( dev_v0, dev_v ); cudaViscosity ( dev_v, dev_v0, 2, &gridDim, &blockDim );

	cudaProjectField ( dev_grid, dev_grid0, dev_u, dev_v, dev_w, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); swap ( dev_v0, dev_v );

	cudaVelAdvect ( dev_u, dev_u0, 1, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );
	cudaVelAdvect ( dev_v, dev_v0, 2, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );
	
	cudaProjectField ( dev_grid, dev_grid0, dev_u, dev_v, dev_w, &gridDim, &blockDim );
*/    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
//		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );
//
//    // Copy output vector from GPU buffer to host memory.
//	if ( cudaMemcpy ( u0, dev_u0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( v0, dev_v0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( w0, dev_w0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//	
//	if ( cudaMemcpy ( u, dev_u, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( v, dev_v, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
//
//	if ( cudaMemcpy ( w, dev_w, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
//		cudaCheckErrors ( "cudaMemcpy was failed" );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function DensityItp
* @author   Orlando Chen
* @date     Dec 18, 2013
* @input    float *dens, GLubyte *data
* @return   NULL
* @bref     Interpolation density and generate a volumetric data set
-----------------------------------------------------------------------------------------------------------
*/
void DensityItp ( float *dens, GLubyte *data )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
	// Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_den, dens, SIM_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );

	kernelInterpolation cudaDevice ( gridDim, blockDim ) ( dev_den, dataset_dev );

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );

	// Return data
	if ( cudaMemcpy ( data, dataset_dev, SIM_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );
};