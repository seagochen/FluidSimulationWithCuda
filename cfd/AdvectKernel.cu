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
* <Last>		Dec 2, 2013
* <File>        AdvectKernel.cu
*/

#ifndef __advect_kernel_cu_
#define __advect_kernel_cu_

#include "cfdHeader.h"

extern void cudaSetBoundary ( float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim );

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

#endif