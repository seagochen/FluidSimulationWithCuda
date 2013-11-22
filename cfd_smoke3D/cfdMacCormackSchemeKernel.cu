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
* <First>       Nov 19, 2013
* <Last>		Nov 20, 2013
* <File>        cfdMacCormackSchemeKernel.cu
*/

#ifndef __cfd_MacCormack_Scheme_Kernel_cu_
#define __cfd_MacCormack_Scheme_Kernel_cu_

#include "cfdHeader.h"

/*
  -----------------------------------------------------------------------------------------------------------
   Define something
  -----------------------------------------------------------------------------------------------------------
*/

#define eqt              ==            /* equal to */
#define and              &&            /* logical and */
#define or               ||            /* logical or */

#define gst_header       0             /* (ghost, halo) the header cell of grid */
#define sim_header       1             /* (actually) the second cell of grid */
#define gst_tailer       Grids_X - 1   /* (ghost, halo) the last cell of grid */
#define sim_tailer       Grids_X - 2   /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_tailer ) \
	if ( j >= sim_header and j <= sim_tailer ) \
	if ( k >= sim_header and k <= sim_tailer ) {

#define EndSimArea() }


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelAddSourceMacCormack
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    float *density_inout, float *velU_inout, float *velV_inout, float *velW_inout
* @return   NULL
* @bref     Add source to simulation grid      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelAddSourceMacCormack ( float *density_inout, float *velU_inout, float *velV_inout, float *velW_inout )
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
			{
				density_inout [ Index (i, j, k) ] += SOURCE * DELTA_TIME;
				velV_inout [ Index (i, j, k) ] = SOURCE * DELTA_TIME;
			}
		}
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAddSourceMacCormack
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *density_inout, float *velU_inout, float *velV_inout, float *velW_inout
* @return   NULL
* @bref     Encapsulation of kernelAddSourceMacCormack    
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaAddSourceMacCormack ( float *density_inout, float *velU_inout, float *velV_inout, float *velW_inout )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	kernelAddSourceMacCormack <<< gridDim, blockDim >>> 
		( density_inout, velU_inout, velV_inout, velW_inout );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function subkernelBoundaryMacCormack
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *density_inout, float *velU_inout, float *velV_inout, float *velW_inout
* @return   NULL
* @bref     Check and set boundary condition      
-----------------------------------------------------------------------------------------------------------
*/
__device__ void subkernelBoundaryMacCormack ( float *density_inout, float *velU_inout, float *velV_inout, float *velW_inout )
{
	// Get index of GPU-thread
	GetIndex ( );

	// Boundary condition
	BeginSimArea();
	{
		// Solve the condition of density
		density_inout [ Index (gst_header, j, k) ] = density_inout [ Index (sim_header, j, k) ]; // bottom side of simulation grid
		density_inout [ Index (gst_tailer, j, k) ] = density_inout [ Index (sim_tailer, j, k) ]; // top side of simulation grid
		density_inout [ Index (i, gst_header, k) ] = density_inout [ Index (i, sim_header, k) ]; // left side of simulation grid
		density_inout [ Index (i, gst_tailer, k) ] = density_inout [ Index (i, sim_tailer, k) ]; // right side of simulation grid
		density_inout [ Index (i, j, gst_header) ] = density_inout [ Index (i, j, sim_header) ]; // front side of simulation grid
		density_inout [ Index (i, j, gst_tailer) ] = density_inout [ Index (i, j, sim_tailer) ]; // back side of simulation grid

		// Solve the condition of sub-flow U
		velU_inout [ Index (gst_header, j, k) ] = -velU_inout [ Index (sim_header, j, k) ]; // bottom side of simulation grid
		velU_inout [ Index (gst_tailer, j, k) ] = -velU_inout [ Index (sim_tailer, j, k) ]; // top side of simulation grid
		velU_inout [ Index (i, gst_header, k) ] =  velU_inout [ Index (i, sim_header, k) ]; // left side of simulation grid
		velU_inout [ Index (i, gst_tailer, k) ] =  velU_inout [ Index (i, sim_tailer, k) ]; // right side of simulation grid
		velU_inout [ Index (i, j, gst_header) ] =  velU_inout [ Index (i, j, sim_header) ]; // front side of simulation grid
		velU_inout [ Index (i, j, gst_tailer) ] =  velU_inout [ Index (i, j, sim_tailer) ]; // back side of simulation grid
		
		// Solve the condition of sub-flow v
		velV_inout [ Index (gst_header, j, k) ] =  velV_inout [ Index (sim_header, j, k) ]; // bottom side of simulation grid
		velV_inout [ Index (gst_tailer, j, k) ] =  velV_inout [ Index (sim_tailer, j, k) ]; // top side of simulation grid
		velV_inout [ Index (i, gst_header, k) ] = -velV_inout [ Index (i, sim_header, k) ]; // left side of simulation grid
		velV_inout [ Index (i, gst_tailer, k) ] =  velV_inout [ Index (i, sim_tailer, k) ]; // right side of simulation grid
		velV_inout [ Index (i, j, gst_header) ] =  velV_inout [ Index (i, j, sim_header) ]; // front side of simulation grid
		velV_inout [ Index (i, j, gst_tailer) ] =  velV_inout [ Index (i, j, sim_tailer) ]; // back side of simulation grid

		// Solve the condition of sub-flow w
		velW_inout [ Index (gst_header, j, k) ] =  velW_inout [ Index (sim_header, j, k) ]; // bottom side of simulation grid
		velW_inout [ Index (gst_tailer, j, k) ] =  velW_inout [ Index (sim_tailer, j, k) ]; // top side of simulation grid
		velW_inout [ Index (i, gst_header, k) ] =  velW_inout [ Index (i, sim_header, k) ]; // left side of simulation grid
		velW_inout [ Index (i, gst_tailer, k) ] =  velW_inout [ Index (i, sim_tailer, k) ]; // right side of simulation grid
		velW_inout [ Index (i, j, gst_header) ] = -velW_inout [ Index (i, j, sim_header) ]; // front side of simulation grid
		velW_inout [ Index (i, j, gst_tailer) ] = -velW_inout [ Index (i, j, sim_tailer) ]; // back side of simulation grid
	}
	EndSimArea();

	// Condition of density
	density_inout [ Index (i, gst_tailer, k) ] *= ANNIHILATION;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function subkernelAdvectDensity
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *den_out, float *den_in, float *velU_in, float *velV_in, float *velW_in, int opt
* @return   NULL
* @bref     Update density state.
* --------- When opt is 0, the prediction is calculated, opt is 1, the checksum value is calculated
-----------------------------------------------------------------------------------------------------------
*/
__device__ void subkernelAdvectDensity ( float *den_out, float *den_in, float *velU_in, float *velV_in, float *velW_in, int opt )
{
	GetIndex();

	BeginSimArea();
	{

	}
	EndSimArea();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function subkernelAdvectFlow
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *vel_out, float *den_in, float *velU_in, float *velV_in, float *velW_in, int opt
* @return   NULL
* @bref     Update a component of flow state
* --------- When opt is 0, the prediction is calculated, opt is 1, the checksum value is calculated
-----------------------------------------------------------------------------------------------------------
*/
__device__ void subkernelAdvectFlow ( float *vel_out, float *den_in, float *velU_in, float *velV_in, float *velW_in, int opt )
{
	GetIndex();

	BeginSimArea();
	{

	}
	EndSimArea();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelPredicateAdvectionMacCormack
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *den_out, float *velU_out, float *velV_out, float *velW_out, 
* --------- float *den_in, float *velU_in, float *velV_in,float *velW_in
* @return   NULL
* @bref     Update flow status      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelPredicateAdvectionMacCormack ( float *den_out, float *velU_out, float *velV_out, float *velW_out, 
										float *den_in, float *velU_in, float *velV_in,float *velW_in )
{
	subkernelAdvectFlow ( velU_out, den_in, velU_in, velV_in, velW_in, 0 );
	subkernelAdvectFlow ( velV_out, den_in, velU_in, velV_in, velW_in, 0 );
	subkernelAdvectFlow ( velW_out, den_in, velU_in, velV_in, velW_in, 0 );
	subkernelAdvectDensity ( den_out, den_in, velU_in, velV_in, velW_in, 0 );
};


/*
-----------------------------------------------------------------------------------------------------------
* @function kernelChecksumAdvectionMacCormack
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *den_out, float *velU_out, float *velV_out, float *velW_out, 
* --------- float *den_in, float *velU_in, float *velV_in,float *velW_in
* @return   NULL
* @bref     Update flow status      
-----------------------------------------------------------------------------------------------------------
*/
__global__ void kernelChecksumAdvectionMacCormack ( float *den_out, float *velU_out, float *velV_out, float *velW_out, 
										float *den_in, float *velU_in, float *velV_in,float *velW_in )
{
	subkernelAdvectFlow ( velU_in, den_out, velU_out, velV_out, velW_out, 1 );
	subkernelAdvectFlow ( velV_in, den_out, velU_out, velV_out, velW_out, 1 );
	subkernelAdvectFlow ( velW_in, den_out, velU_out, velV_out, velW_out, 1 );
	subkernelAdvectDensity ( den_in, den_out, velU_out, velV_out, velW_out, 1 );
};


__global__ void kernelFinalAdvectionMacCormack ( float *den_out, float *velU_out, float *velV_out, float *velW_out, 
										float *den_in, float *velU_in, float *velV_in,float *velW_in )
{
	GetIndex ( );

	den_out [ Index(i,j,k) ] = 0.5f * ( den_out [ Index(i,j,k) ] + den_in [ Index(i,j,k) ] );
	velU_out [ Index(i,j,k) ] = 0.5f * ( velU_out [ Index(i,j,k) ] + velU_in [ Index(i,j,k) ] );
	velV_out [ Index(i,j,k) ] = 0.5f * ( velV_out [ Index(i,j,k) ] + velV_in [ Index(i,j,k) ] );
	velW_out [ Index(i,j,k) ] = 0.5f * ( velW_out [ Index(i,j,k) ] + velW_in [ Index(i,j,k) ] );

	// Check boundary condition
	subkernelBoundaryMacCormack ( den_out, velU_out, velV_out, velW_out );
};

/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAdvectMacCormack
* @author   Orlando Chen
* @date     Nov 22, 2013
* @input    float *den_out, float *velU_out, float *velV_out, float *velW_out, 
* --------- float *den_in, float *velU_in, float *velV_in,float *velW_in
* @return   NULL
* @bref     Update flow status      
-----------------------------------------------------------------------------------------------------------
*/
__host__ void cudaAdvectMacCormack ( float *den_out, float *velU_out, float *velV_out, float *velW_out, 
										float *den_in, float *velU_in, float *velV_in,float *velW_in )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	// Calculate the advection
	kernelPredicateAdvectionMacCormack <<<gridDim, blockDim>>> 
		( den_out, velU_out, velV_out, velW_out, den_in, velU_in, velV_in, velW_in );
	kernelChecksumAdvectionMacCormack  <<<gridDim, blockDim>>>
		( den_out, velU_out, velV_out, velW_out, den_in, velU_in, velV_in, velW_in );
};

/*
-----------------------------------------------------------------------------------------------------------
* @function MacCormackSchemeSolver
* @author   Orlando Chen
* @date     Nov 19, 2013
* @input    u, v, w, u0, v0, w0, grid, grid0
* @return   NULL
* @bref     Calculate the advection of flow by MackCormack Scheme
-----------------------------------------------------------------------------------------------------------
*/
__host__ void MacCormackSchemeSolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0, float *grid, float *grid0 )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
	if ( cudaMemcpy ( dev_grid, grid, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( dev_grid0, grid0, SIM_SIZE * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

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

	// Launch kernels
	// Add source to background for further simulation
	cudaAddSourceMacCormack (dev_den0, dev_u0, dev_v0, dev_w0);
	cudaAdvectMacCormack (dev_den, dev_u, dev_v, dev_w, dev_den0, dev_u0, dev_v0, dev_w0);
	// ...
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	if ( cudaDeviceSynchronize ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceSynchronize was failed" );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( grid, dev_grid, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( grid0, dev_grid0, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

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
};

/*
  -----------------------------------------------------------------------------------------------------------
   Undefine something
  -----------------------------------------------------------------------------------------------------------
*/

#undef eqt   /* equal to */
#undef and   /* logical and */
#undef or    /* logical or */

#undef gst_header  /* (ghost, halo) the header cell of grid */
#undef sim_header  /* (actually) the second cell of grid */
#undef gst_tailer  /* (ghost, halo) the last cell of grid */
#undef sim_tailer  /* (actually) the second last cell of grid */

#endif