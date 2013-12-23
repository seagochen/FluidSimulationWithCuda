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
* <First>       Dec 12, 2013
* <Last>		Dec 23, 2013
* <File>        kernel.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SGE\SGUtils.h>
#include "fluidsim.h"

using namespace sge;
using namespace std;


#pragma region free, zero, and allocate memory

void FluidSim::FreeResourcePtrs ( void )
{
	// Release ptr of host
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( host_list [ i ] ) SAFE_FREE_PTR ( host_list [ i ] );
	}
	host_list.empty ( );

	// Release ptr of CUDA
	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );
}

void FluidSim::ZeroData ( void )
{
	for ( int i = 0; i < SIM_SIZE; i++ )
	{
		host_u [ i ] = 0.f;
		host_v [ i ] = 0.f;
		host_w [ i ] = 0.f;
		host_u0 [ i ] = 0.f;
		host_v0 [ i ] = 0.f;
		host_w0 [ i ] = 0.f;
		host_den [ i ] = 0.f;
		host_den0 [ i ] = 0.f;
	}
}

SGRUNTIMEMSG FluidSim::AllocateResourcePtrs ( void )
{
	/// Choose which GPU to run on, change this on a multi-GPU system. ///
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices" );

	/// Allocate memory on host ///
	for ( int i = 0; i < HostListNum; i++ )
	{
		static float *ptr;
		ptr = (float*) malloc ( SIM_SIZE * sizeof(float) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
			return SG_RUNTIME_FALSE;
		}
	}

	/// Allocate memory on GPU devices ///
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static float *ptr;
		if ( cudaMalloc( (void **) &ptr, SIM_SIZE * sizeof(float) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	// Finally
	return SG_RUNTIME_OK;
}  

#pragma endregion


#pragma region buffer operations

__global__ void kernelPickData ( float *grid, unsigned char *data )
{
	GetIndex();

	float temp = grid [ Index (i, j, k) ];
	if ( temp > 256.f ) temp = 256.f;
	else if ( temp < 0.f ) temp = 0.f;

	data [ Index (i, j, k) ] = (unsigned char) temp;
};

__global__ void kernelZeroBuffer ( float *grid )
{
	GetIndex ();

	grid [ Index(i,j,k) ] = 0.f;
};

__global__ void kernelCopyBuffer ( float *grid_out, const float *grid_in )
{
	GetIndex ();
	grid_out [ Index(i,j,k) ] = grid_in [ Index(i,j,k) ];
};

__global__ void kernelSwapBuffer ( float *grid0, float *grid1 )
{
	GetIndex ();
	float temp = grid0 [ Index(i,j,k) ];
	grid0 [ Index(i,j,k) ] = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = temp;
};

#pragma endregion


__global__ void kernelAddSource ( float *dens, float *u, float *v, float*w )
{
	GetIndex();

	const int half = Grids_X / 2;
	if ( i > half - 10 && i < half + 10 ) if ( j < 2 ) if ( k > half - 10 && k < half + 10 )
	{
		if ( dens != NULL )
			dens [ Index(i,j,k) ] = SOURCE * DELTA_TIME;
		if ( v != NULL )
			v [ Index(i,j,k) ] = SOURCE * DELTA_TIME;
	}
};


__device__ void subCheckBoundary ( float *grid, const int condition )
{
/**
* condition can be one of the following
* 0 -------- density solver
* 1 -------- velocity u solver
* 2 -------- velocity v solver
* 3 -------- velocity w solver
*/
	GetIndex ();

	BeginSimArea ();
	{
		switch ( condition )
		{
		case 0: // density
			/// density, simply copy the data ///
			grid [ Index(gst_header, j, k) ] = grid [ Index(sim_header, j, k) ];
			grid [ Index(i, gst_header, k) ] = grid [ Index(i, sim_header, k) ];
			grid [ Index(i, j, gst_header) ] = grid [ Index(i, j, sim_header) ];
			grid [ Index(gst_trailer, j, k) ] = grid [ Index(sim_trailer, j, k) ];
			grid [ Index(i, gst_trailer, k) ] = grid [ Index(i, sim_trailer, k) ];
			grid [ Index(i, j, gst_trailer) ] = grid [ Index(i, j, sim_trailer) ];
			break;

		case 1: // velocity u
			/// plane ik will keep the same as vector u ///
			grid [ Index(i, gst_header, k) ] = grid [ Index(i, sim_header, k) ];
			grid [ Index(i, gst_trailer, k) ] = grid [ Index(i, sim_trailer, k) ];
			/// as vector u is perpendicular to plane jk ///
			if ( grid [ Index(sim_header, j, k) ] > 0 )
				grid [ Index(gst_header, j, k) ] = grid [ Index(sim_header, j, k) ];
			else
				grid [ Index(gst_header, j, k) ] = -grid [ Index(sim_header, j, k) ];
			if ( grid [ Index(sim_trailer, j, k) ] > 0 )
				grid [ Index(gst_trailer, j, k) ] = -grid [ Index(sim_trailer, j, k) ];
			else
				grid [ Index(gst_trailer, j, k) ] = grid [ Index(sim_trailer, j, k) ];
			/// plane ij is the same as plane ik ///
			grid [ Index(i, j, gst_header) ] = grid [ Index(i, j, sim_header) ];			
			grid [ Index(i, j, gst_trailer) ] = grid [ Index(i, j, sim_trailer) ];
			break;

		case 2: // velocity v
			/// vector v is perpendicular to plane ik ///
			if ( grid [ Index(i, sim_header, k) > 0 ] )
				grid [ Index(i, gst_header, k) ] = grid [ Index(i, sim_header, k) ];
			else
				grid [ Index(i, gst_header, k) ] = -grid [ Index(i, sim_header, k) ];
			if ( grid [ Index(i, sim_trailer, k) ] > 0 )
				grid [ Index(i, gst_trailer, k) ] = -grid [ Index(i, sim_trailer, k) ];
			else 
				grid [ Index(i, gst_trailer, k) ] = grid [ Index(i, sim_trailer, k) ];
			/// vector v parallel to the plane ij, jk ///
			grid [ Index(gst_header, j, k) ] = grid [ Index(sim_header, j, k) ];
			grid [ Index(i, j, gst_header) ] = grid [ Index(i, j, sim_header) ];
			grid [ Index(gst_trailer, j, k) ] = grid [ Index(sim_trailer, j, k) ];
			grid [ Index(i, j, gst_trailer) ] = grid [ Index(i, j, sim_trailer) ];
			break;

		case 3: // velocity w
			/// vector w is perpendicular to plane ij ///
			if ( grid [ Index(i, j, sim_header) ] > 0 )
				grid [ Index(i, j, gst_header) ] = grid [ Index(i, j, sim_header) ];
			else
				grid [ Index(i, j, gst_header) ] = -grid [ Index(i, j, sim_header) ];
			if ( grid [ Index(i, j, sim_trailer) ] > 0 )
				grid [ Index(i, j, gst_trailer) ] = -grid [ Index(i, j, sim_trailer) ];
			else
				grid [ Index(i, j, gst_trailer) ] = grid [ Index(i, j, sim_trailer) ];
			/// vector w is parallel to the plane jk, ik ///
			grid [ Index(gst_header, j, k) ] = grid [ Index(sim_header, j, k) ];
			grid [ Index(i, gst_header, k) ] = grid [ Index(i, sim_header, k) ];
			grid [ Index(gst_trailer, j, k) ] = grid [ Index(sim_trailer, j, k) ];
			grid [ Index(i, gst_trailer, k) ] = grid [ Index(i, sim_trailer, k) ];
			break;

		default:
			break;
		}
	}
	EndSimArea ();

		/// eight corner cells ///
		grid [ Index(gst_header, gst_header, gst_header) ] = (          // at (0, 0, 0)
			grid [ Index(sim_header, gst_header, gst_header) ] +          // --right
			grid [ Index(gst_header, sim_header, gst_header) ] +          // --up
			grid [ Index(gst_header, gst_header, sim_header) ] ) / 3.f;   // --in
		grid [ Index(gst_header, gst_trailer, gst_header) ] = (         // at (0, 1, 0)
			grid [ Index(gst_header, sim_trailer, gst_header) ] +         // --down
			grid [ Index(sim_header, gst_trailer, gst_header) ] +         // --right
			grid [ Index(gst_header, gst_trailer, sim_header) ]) / 3.f;   // --in
		grid [ Index(gst_trailer, gst_trailer, gst_header) ] = (        // at (1, 1, 0)
			grid [ Index(sim_trailer, gst_trailer, gst_header) ] +        // --left
			grid [ Index(gst_trailer, sim_trailer, gst_header) ] +        // --down
			grid [ Index(gst_trailer, gst_trailer, sim_header) ]) / 3.f;  // --in
		grid [ Index(gst_trailer, gst_header, gst_header) ] = (         // at (1, 0, 0)
			grid [ Index(sim_trailer, gst_header, gst_header) ] +         // --left 
			grid [ Index(gst_trailer, sim_header, gst_header) ] +         // --up
			grid [ Index(gst_trailer, gst_header, sim_header) ] ) / 3.f;  // --in
		grid [ Index(gst_header, gst_header, gst_trailer) ] = (         // at (0, 0, 1)
			grid [ Index(gst_header, gst_header, sim_trailer) ] +         // --out
			grid [ Index(gst_header, sim_header, gst_trailer) ] +         // --up
			grid [ Index(sim_header, gst_header, gst_trailer) ]) / 3.f;   // --right
		grid [ Index(gst_header, gst_trailer, gst_trailer) ] = (        // at (0, 1, 1)
			grid [ Index(gst_header, gst_trailer, sim_trailer) ] +        // --out
			grid [ Index(gst_header, sim_trailer, gst_trailer) ] +        // --down
			grid [ Index(sim_header, gst_trailer, gst_trailer) ]) / 3.f;  // --right
		grid [ Index(gst_trailer, gst_trailer, gst_trailer) ] = (       // at (1, 1, 1)
			grid [ Index(sim_trailer, gst_trailer, gst_trailer) ] +       // --left
			grid [ Index(gst_trailer, sim_trailer, gst_trailer) ] +       // --down
			grid [ Index(gst_trailer, gst_trailer, sim_trailer) ]) / 3.f; // --out
		grid [ Index(gst_trailer, gst_header, gst_trailer) ] = (        // at (1, 0, 1)
			grid [ Index(gst_trailer, gst_header, sim_trailer) ] +        // --out 
			grid [ Index(gst_trailer, sim_header, gst_trailer) ] +        // --up 
			grid [ Index(sim_trailer, gst_header, gst_trailer) ]) / 3.f;  // --left
};


__global__ void kernelCheckBoundary ( float *dens, float *velU, float *velV, float *velW )
{
	if ( dens != NULL )
		subCheckBoundary ( dens, 0 );
	if ( velU != NULL )
		subCheckBoundary ( velU, 1 );
	if ( velV != NULL )
		subCheckBoundary ( velV, 2 );
	if ( velW != NULL )
		subCheckBoundary ( velW, 3 );
}


__device__ float subInterpolation ( float v0, float v1, float w0, float w1 )
{
	return v0 * w0 + v1 * w1;
};


__global__ void kernelAdvect ( float *grid_out, float const *grid_in, float const *u_in, float const *v_in, float const *w_in )
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
		float tempi0j0 = subInterpolation ( grid_in [ Index (i0, j0, k0) ], grid_in [ Index (i0, j0, k1) ], w0, w1 );
		float tempi0j1 = subInterpolation ( grid_in [ Index (i0, j1, k0) ], grid_in [ Index (i0, j1, k1) ], w0, w1 );
		float tempi1j0 = subInterpolation ( grid_in [ Index (i1, j0, k0) ], grid_in [ Index (i1, j0, k1) ], w0, w1 );
		float tempi1j1 = subInterpolation ( grid_in [ Index (i1, j1, k0) ], grid_in [ Index (i1, j1, k1) ], w0, w1 );

		// 對點<latex>{P}'</latex>，v方向做插值計算
		float tempi0   = subInterpolation ( tempi0j0, tempi0j1, v0, v1 );
		float tempi1   = subInterpolation ( tempi1j0, tempi1j1, v0, v1 );

		// 對點<latex>{P}'</latex>，u方向做插值計算, 並獲得最終結果
		grid_out [ Index(i, j, k) ] = subInterpolation ( tempi0, tempi1, u0, u1 );

//		grid_out [ Index(i0,j0,k0) ] = ( grid_in [ Index(i0,j0,k0) ] + 
//			grid_in [ Index(i0+1,j0,k0) ] + grid_in [ Index(i0-1,j0,k0) ] + 
//			grid_in [ Index(i0,j0+1,k0) ] + grid_in [ Index(i0,j0-1,k0) ] +
//			grid_in [ Index(i0,j0,k0+1) ] + grid_in [ Index(i0,j0,k0-1) ] ) / 7.f;
	}
	EndSimArea();

};


__device__ float subDivergence ( const float *grid_in, int i, int j, int k )
{
	/**
	* As delta x, delta y, and delta z is strictly equal to 1 
	* which results the computation is simplified as the following 
	*/
	return
		grid_in[Index(i-1, j, k)] + grid_in[Index(i+1, j, k)] + 
		grid_in[Index(i, j-1, k)] + grid_in[Index(i, j+1, k)] +
		grid_in[Index(i, j, k-1)] + grid_in[Index(i, j, k+1)];
};


__global__ void kernelDiffuse ( float *grid_out, const float *grid_in )
{
	GetIndex ();

	float ratio = DELTA_TIME * DIFFUSION * SimArea_X * SimArea_X;
	float div   = 1.f + 6.f * ratio;

	BeginSimArea ();
	{
//		grid_out [ Index(i, j, k) ] = ( grid_in [ Index(i, j, k) ] + ratio * ( subDivergence (grid_out, i, j, k) )) / div;
		grid_out [ Index(i,j,k) ] = ( grid_in [ Index(i,j,k) ] + ratio * ( subDivergence (grid_in, i, j, k) )) / div;
	}
	EndSimArea ();
};


__global__ void kernelVisocity ( float *grid_out, float const *grid_in )
{
	GetIndex ( );
	float ratio = DELTA_TIME * VISCOSITY * SimArea_X * SimArea_X;
	float div   = 1.f + 6.f * ratio;

	BeginSimArea ( );
	{
//		grid_out [ Index(i, j, k) ] = ( grid_in [ Index(i, j, k) ] + ratio * ( subDivergence (grid_out, i, j, k) )) / div;
		grid_out [ Index(i, j, k) ] = ( grid_in [ Index(i, j, k) ] + ratio * ( subDivergence (grid_in, i, j, k) )) / div;
	}
	EndSimArea ( );
};


__global__ void kernelDivergence ( float *grad_out, float *proj_out, float const *u_in, float const *v_in, float const *w_in )
{
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


__global__ void kernelConservField ( float *proj_out, float const *grad_in, float const *u_in, float const *v_in, float const *w_in )
{
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


__global__ void kernelProjectVelocity ( float *u_out, float *v_out, float *w_out, float const *grad_in, float const *proj_in )
{
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


FluidSim::FluidSim ( fluidsim *fluid )
{
#pragma region allocate host memory and device buffer
	size_t size = fluid->nVolDepth * fluid->nVolHeight * fluid->nVolWidth;
	data = (GLubyte *) malloc ( size * sizeof(GLubyte) );
	if ( cudaMalloc( (void**) &dev_data, SIM_SIZE * sizeof(unsigned char) ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!" );
		cudaFree (dev_data);
		SAFE_FREE_PTR ( data );
		exit (1);
	}
	if ( AllocateResourcePtrs () != SG_RUNTIME_OK )
	{
		FreeResourcePtrs ();
		SAFE_FREE_PTR ( data );
		exit (1);
	}
#pragma endregion
	
	first = false;

	cout << "fluid simulation ready, zero the data and preparing the stage now" << endl;
	ZeroData ();
	
#pragma region do add source and velocity first
	cudaDeviceDim3D ();
	
	kernelAddSource cudaDevice ( gridDim, blockDim ) ( dev_den, dev_u, dev_v, dev_w );
	
	if ( cudaDeviceSynchronize () != cudaSuccess )
		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );

	/// We do not retrieve the data right now, just keep them on buffer for next step
#pragma endregion
};


void FluidSim::DensitySolver ( void )
{
	/// Define the computing unit size ///
	cudaDeviceDim3D ();

	/// Zero origin buffer first ///
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_den0 );
	/// Advection ///
	kernelAdvect cudaDevice(gridDim, blockDim) ( dev_den0, dev_den, dev_u, dev_v, dev_w );
	/// Check boundary condtion ///
	kernelCheckBoundary cudaDevice(gridDim, blockDim) ( dev_den0, NULL, NULL, NULL );
	/// Copy the buffer ///
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_den, dev_den0 );
	// Diffusion
	kernelDiffuse cudaDevice(gridDim, blockDim) ( dev_den0, dev_den );
	/// Check boundary condtion ///
	kernelCheckBoundary cudaDevice(gridDim, blockDim) ( dev_den0, NULL, NULL, NULL );
	/// Swap buffer ///
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_den, dev_den0 );
	/// Zero origin buffer ///
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_den0 );
};


void FluidSim::VelocitySolver ( void )
{
	/// Define the computing unit size ///
	cudaDeviceDim3D ();

	/// Zero origin buffer first ///
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_u0 );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_v0 );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_w0 );
	/// Visocity ///
	kernelVisocity cudaDevice(gridDim, blockDim) ( dev_u0, dev_u ); 
	kernelVisocity cudaDevice(gridDim, blockDim) ( dev_v0, dev_v );
	kernelVisocity cudaDevice(gridDim, blockDim) ( dev_w0, dev_w );
	kernelCheckBoundary cudaDevice(gridDim, blockDim) ( NULL, dev_u0, dev_v0, dev_w0 );
	/// Copy buffer ///
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_u, dev_u0 );
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_v, dev_v0 );
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_w, dev_w0 );
	/// Zero buffer ///
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_u0 ); 
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_v0 );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_w0 ); 
	/// Project field ///
	kernelDivergence cudaDevice(gridDim, blockDim) ( dev_grid, dev_grid0, dev_u, dev_v, dev_w );
	kernelConservField cudaDevice(gridDim, blockDim) (dev_grid, dev_grid0, dev_u, dev_v, dev_w );
	kernelProjectVelocity cudaDevice(gridDim, blockDim) ( dev_u, dev_v, dev_w, dev_grid, dev_grid0 );
	/// Velocity update ///
	kernelAdvect cudaDevice(gridDim, blockDim) ( dev_u0, dev_u, dev_u, dev_v, dev_w );
	kernelAdvect cudaDevice(gridDim, blockDim) ( dev_v0, dev_v, dev_u, dev_v, dev_w );
	kernelAdvect cudaDevice(gridDim, blockDim) ( dev_w0, dev_w, dev_u, dev_v, dev_w );
	kernelCheckBoundary cudaDevice(gridDim, blockDim) ( NULL, dev_u0, dev_v0, dev_w0 );
	/// Copy buffer ///
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_u, dev_u0 );
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_v, dev_v0 );
	kernelCopyBuffer cudaDevice(gridDim, blockDim) ( dev_w, dev_w0 );
	/// Project field again ///
	kernelDivergence cudaDevice(gridDim, blockDim) ( dev_grid, dev_grid0, dev_u, dev_v, dev_w );
	kernelConservField cudaDevice(gridDim, blockDim) (dev_grid, dev_grid0, dev_u, dev_v, dev_w );
	kernelProjectVelocity cudaDevice(gridDim, blockDim) ( dev_u, dev_v, dev_w, dev_grid, dev_grid0 );
	/// Zero buffer ///
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_u0 ); 
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_v0 );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_w0 ); 
};


void FluidSim::FluidSimSolver ( fluidsim *fluid )
{
	// Solving the fluid simulation
	VelocitySolver ();
	DensitySolver ();

	// Define the computing unit size
	cudaDeviceDim3D ();
   
	kernelPickData cudaDevice ( gridDim, blockDim ) ( dev_den, dev_data );

	if ( cudaDeviceSynchronize () != cudaSuccess )
		cudaCheckErrors ( "cudaDeviceSynchronize was failed" );
	
	if ( cudaMemcpy ( data, dev_data, sizeof(unsigned char) * SIM_SIZE, cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );
	
	if ( cudaMemcpy ( host_den, dev_den, sizeof(float) * SIM_SIZE, cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckErrors ( "cudaMemcpy was failed" );

	fluid->ptrData = data;
};