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
* <File>        SetBoundaryKernel.cu
*/

#ifndef __set_boundary_kernel_cu_
#define __set_boundary_kernel_cu_

#include "cfdHeader.h"

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

#endif