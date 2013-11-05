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
* <Last>		Nov 5, 2013
* <File>        cfdKernel.cu
*/

#ifndef __cfd_2DKernel_cu_
#define __cfd_2DKernel_cu_

#include "macroDef.h"
#include "cudaHelper.h"

using namespace std;

#define cudaKernel3DIndex(i, j, k) \
	i = blockIdx.x * blockDim.x + threadIdx.x; \
	j = blockIdx.y * blockDim.y + threadIdx.y; \
	k = 0; \
	cudaTrans2DTo3D(i, j, k, Grids_X); \

#define cudaKernel2DIndex(i, j) \
	i = blockIdx.x * blockDim.x + threadIdx.x; \
	j = blockIdx.y * blockDim.y + threadIdx.y; \


__global__ void add_source_kernel ( float *ptr_out )
{
	// Get index of GPU-thread
	int i, j, k;
	cudaKernel3DIndex ( i, j, k );

	if ( i == Grids_X / 2 && j ==  Grids_X / 2 )
	{
		int ind = cudaIndex3D(i, j, k, Grids_X);
		ptr_out[ind] += DELTA_TIME * 1.f;
	}
};


__global__ void set_bnd_kernel ( float *grid_out, int boundary )
{
#define is ==
#define gst0    0
#define gstl    Grids_X - 1
#define real0   1
#define reall   SimArea_X

	// Get index of GPU-thread
	int i, j, k;
	cudaKernel3DIndex(i, j, k);

	// Boundary condition
	if ( i >= 1 && i <= SimArea_X ) if ( j >= 1 && j <= SimArea_X ) if ( k >= 1 && k <= SimArea_X )
	{
		// Slove line (ghost-0, y, z)
		grid_out [ cudaIndex3D ( gst0, j, k, Grids_X ) ] = boundary is 1 ? -grid_out [ cudaIndex3D ( real0, j, k, Grids_X ) ] : grid_out [ cudaIndex3D ( real0, j, k, Grids_X ) ];
		// Slove line (ghost-1, y, z)
		grid_out [ cudaIndex3D ( gstl, j, k, Grids_X ) ] = boundary is 1 ? -grid_out [ cudaIndex3D ( reall, j, k, Grids_X ) ] : grid_out [ cudaIndex3D ( reall, j, k, Grids_X ) ];
		// Slove line (x, ghost-0, z)
		grid_out [ cudaIndex3D ( i, gst0, k, Grids_X ) ] = boundary is 2 ? -grid_out [ cudaIndex3D ( i, real0, k, Grids_X ) ] : grid_out [ cudaIndex3D ( i, real0, k, Grids_X ) ];
		// Slove line (x, ghost-1, z)
		grid_out [ cudaIndex3D ( i, gstl, k, Grids_X ) ] = boundary is 2 ? -grid_out [ cudaIndex3D ( i, reall, k, Grids_X ) ] : grid_out [ cudaIndex3D ( i, reall, k, Grids_X ) ];
		// Slove line (x, y, ghost-0)
		grid_out [ cudaIndex3D ( i, j, gst0, Grids_X ) ] = boundary is 1 ? -grid_out [ cudaIndex3D ( i, j, real0, Grids_X ) ] : grid_out [ cudaIndex3D ( i, j, real0, Grids_X ) ];
		// Slove line (x, y, ghost-1)
		grid_out [ cudaIndex3D ( i, j, gstl, Grids_X ) ] = boundary is 1 ? -grid_out [ cudaIndex3D ( i, j, reall, Grids_X ) ] : grid_out [ cudaIndex3D ( i, j, reall, Grids_X ) ];
	}
	// Slove ghost cell (ghost-0, ghost-0, ghost-0)
	//grid_out[cudaIndex3D(gst0, gst0, gst0, Grids_X)] = 0.5f * ( grid_out[cudaIndex3D(real0, gst0, Grids_X)]  + grid_out[cudaIndex2D(0, 1, Grids_X)] );
	// Slove ghost cell (0, 65)
	//grid_out[cudaIndex2D(gst0, 65, Grids_X)] = 0.5f * ( grid_out[cudaIndex2D(1, 65, Grids_X)] + grid_out[cudaIndex2D(0, 64, Grids_X)] );
	// Slove ghost cell (65, 0)
	//grid_out[cudaIndex2D(65, gst0, Grids_X)] = 0.5f * ( grid_out[cudaIndex2D(64, 0, Grids_X)] + grid_out[cudaIndex2D(65, 1, Grids_X)] );
	// Slove ghost cell (65, 65)
	//grid_out[cudaIndex2D(65, 65, Grids_X)] = 0.5f * ( grid_out[cudaIndex2D(64, 65, Grids_X)] + grid_out[cudaIndex2D(65, 64, Grids_X)]);

#undef is
#undef gst0
#undef gstl
#undef real0
#undef reall
}


__global__ void lin_solve_kernel ( float *grid_inout, float *grid0_in, int boundary, float a, float c )
{
	// Get index of GPU-thread
	int i, j, k;
	cudaKernel3DIndex ( i, j, k );

	if ( i >= 1 && i <= SimArea_X ) if ( j >= 1 && j <= SimArea_X ) if ( k >= 1 && k <= SimArea_X )
	{	
		grid_inout [ cudaIndex3D ( i, j, k, Grids_X ) ] = ( grid0_in [ cudaIndex3D ( i, j, k, Grids_X ) ] + a * ( grid_inout [ cudaIndex3D ( i-1, j, k, Grids_X ) ] + 
			grid_inout [ cudaIndex3D ( i+1, j, k, Grids_X ) ] + grid_inout [ cudaIndex3D ( i, j-1, k, Grids_X ) ] + grid_inout [ cudaIndex3D ( i, j+1, k, Grids_X ) ] ) ) / c;	
	}
}


__global__ void advect_kernel ( float *density_out, float *density0_in, float *u_in, float *v_in, float *w_in, float dt0 )
{
	// Get index of GPU-thread
	int i, j, k;
	cudaKernel3DIndex ( i, j, k );

	if ( i >= 1 && i <= SimArea_X ) if ( j >= 1 && j <= SimArea_X ) if ( k >= 1 && k <= SimArea_X )
	{
		int x = i - dt0 * u_in [ cudaIndex3D ( i, j, k, Grids_X ) ];
		int y = j - dt0 * v_in [ cudaIndex3D ( i, j, k, Grids_X ) ];
		int z = k - dt0 * w_in [ cudaIndex3D ( i, j, k, Grids_X ) ];

		if ( x < 0.5f ) x = 0.5f;
		if ( x > SimArea_X + 0.5f ) x = SimArea_X + 0.5f;

		int i0 = (int) x; 
		int i1 = i0+1;
		
		if ( y < 0.5f ) y = 0.5f;
		if ( y > SimArea_X + 0.5f ) y = SimArea_X + 0.5f;
		
		int j0 = ( int ) y;
		int j1 = j0 + 1;

		if ( z < 0.5f ) z = 0.5f;
		if ( z > SimArea_X + 0.5f ) z = SimArea_X + 0.5f;

		int k0 = ( int ) z;
		int k1 = k0 + 1;

		int s1 = x - i0;
		int s0 = 1 - s1;
		int t1 = y - j0;
		int t0 = 1 - t1;

		// TODO ...
		density_out [ cudaIndex3D ( i, j, k, Grids_X ) ] = s0 * ( t0 * density0_in [ cudaIndex3D( i0, j0, k0, Grids_X ) ] +
			t1 * density0_in [ cudaIndex3D ( i0, j1, k0, Grids_X ) ] ) + s1 * ( t0 * density0_in [ cudaIndex3D ( i1, j0, k0, Grids_X ) ] + 
			t1 * density0_in [ cudaIndex3D ( i1, j1, k0, Grids_X ) ] );
	}
};


__global__ void project_kernel_pt1 ( float * u, float * v, float * p, float * div )
{
	// Get index of GPU-thread
	int i, j, k;
	cudaKernel3DIndex ( i, j, k );
	
	// TODO ...
	if ( i >= 1 && i <= SimArea_X ) if ( j >= 1 && j <= SimArea_X ) if ( k >= 1 && k <= SimArea_X )
	{
		div [ cudaIndex3D ( i, j, k, Grids_X ) ] = -0.5f * ( u[ cudaIndex3D ( i+1, j, k, Grids_X ) ] - u [ cudaIndex3D ( i-1, j, k, Grids_X ) ] +
			v [ cudaIndex3D ( i, j+1, k, Grids_X ) ] - v [ cudaIndex3D ( i, j-1, k, Grids_X ) ] ) / SimArea_X;
		p [ cudaIndex3D ( i, j, k, Grids_X ) ] = 0;
	}
}


__global__ void project_kernel_pt2 ( float * u, float * v, float * p, float * div )
{
	// Get index of GPU-thread
	int i, j, k;
	cudaKernel3DIndex ( i, j, k );
	
	// TODO ...
	if ( i >= 1 && i <= SimArea_X ) if ( j >= 1 && j <= SimArea_X ) if ( k >= 1 && k <= SimArea_X )
	{
		u [ cudaIndex3D( i, j, k, Grids_X ) ] -= 0.5f * SimArea_X * ( p [ cudaIndex3D( i+1, j, k, Grids_X ) ] - p [ cudaIndex3D ( i-1, j, k, Grids_X ) ] );
		v [ cudaIndex3D( i, j, k, Grids_X ) ] -= 0.5f * SimArea_X * ( p [ cudaIndex3D( i, j+1, k, Grids_X ) ] - p [ cudaIndex3D ( i, j-1, k, Grids_X ) ] );
	}
}


void cuda_add_source ( float *grid, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	add_source_kernel cudaDevice ( *gridDim,  *blockDim ) ( grid );
};


void cuda_lin_solve ( float *grid, float *grid0, int boundary, float a, float c, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	for ( int i = 0; i < 20; i ++ )
	{
		lin_solve_kernel cudaDevice ( *gridDim,  *blockDim ) ( grid, grid0, boundary, a, c );
	}
	set_bnd_kernel cudaDevice ( *gridDim,  *blockDim )  ( grid, boundary );
}


void cuda_diffuse ( float *grid, float *grid0, int boundary, float diff, dim3 *gridDim, dim3 *blockDim )
{
	float a = DELTA_TIME * diff * SimArea_X * SimArea_X;
	cuda_lin_solve ( grid, grid0, boundary, a, 1 + 4 * a, gridDim, blockDim );
}


void cuda_advect ( float *density, float *density0, float *u, float *v,  int boundary, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	float dt0 = DELTA_TIME * SimArea_X;

	advect_kernel cudaDevice ( *gridDim,  *blockDim ) ( density, density0, u, v, dt0 );
	set_bnd_kernel cudaDevice ( *gridDim,  *blockDim ) ( density, boundary );
}


void cuda_project ( float * u, float * v, float * p, float * div, dim3 *gridDim, dim3 *blockDim )
{
	project_kernel_pt1  cudaDevice ( *gridDim,  *blockDim )  ( u, v, p, div );
	set_bnd_kernel  cudaDevice ( *gridDim,  *blockDim )  ( div, 0 ); 
	set_bnd_kernel  cudaDevice ( *gridDim,  *blockDim )  ( p, 0 );
	lin_solve_kernel  cudaDevice ( *gridDim,  *blockDim )  ( p, div, 0, 1, 4 );
	project_kernel_pt2  cudaDevice ( *gridDim,  *blockDim )  ( u, v, p, div );
	set_bnd_kernel  cudaDevice ( *gridDim,  *blockDim )  ( u, 1 );
	set_bnd_kernel  cudaDevice ( *gridDim,  *blockDim )  ( v, 2 );
}


void dens_step ( float *grid, float *grid0, float *u, float *v )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	size_t size = SIZE;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy ( dev_grid, grid, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_grid0, grid0, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit(0);
    }

	cudaStatus = cudaMemcpy ( dev_u, u, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit(0);    
	}

	cudaStatus = cudaMemcpy ( dev_v, v, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit(0);
    }


	cuda_add_source ( dev_grid, &gridDim, &blockDim );
	swap ( dev_grid0, dev_grid ); cuda_diffuse ( dev_grid, dev_grid0, 0, DIFFUSION, &gridDim, &blockDim );
	swap ( dev_grid0, dev_grid ); cuda_advect ( dev_grid, dev_grid0, dev_u, dev_v, 0, &gridDim, &blockDim );
	
	
	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError ( );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "CUDA encountered an error, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize ( );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaDeviceSynchronize was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy ( grid, dev_grid, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( grid0, dev_grid0, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
	}
	
	cudaStatus = cudaMemcpy ( u, dev_u, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
	}

	cudaStatus = cudaMemcpy ( v, dev_v, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
	}
}


void vel_step ( float * u, float * v, float * u0, float * v0 )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	size_t size = SIZE;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy ( dev_u0, u0, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_v0, v0, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) ); 
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_u, u, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_v, v, size * sizeof ( float ), cudaMemcpyHostToDevice );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }


	cuda_add_source ( dev_u, &gridDim, &blockDim ); cuda_add_source ( dev_v, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); cuda_diffuse ( dev_u, dev_u0, 1, VISCOSITY, &gridDim, &blockDim );
	swap ( dev_v0, dev_v ); cuda_diffuse ( dev_v, dev_v0, 2, VISCOSITY, &gridDim, &blockDim );
	cuda_project ( dev_u, dev_v, dev_u0, dev_v0, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); swap ( dev_v0, dev_v );
	cuda_advect ( dev_u, dev_u0, dev_u0, dev_v0, 1, &gridDim, &blockDim );
	cuda_advect ( dev_v, dev_v0, dev_u0, dev_v0, 2, &gridDim, &blockDim );
	cuda_project ( dev_u, dev_v, dev_u0, dev_v0, &gridDim, &blockDim );


	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError ( );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "CUDA encountered an error, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize ( );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaDeviceSynchronize was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy ( u0, dev_u0, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( v0, dev_v0, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }
	
	cudaStatus = cudaMemcpy ( u, dev_u, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( v, dev_v, size * sizeof ( int ), cudaMemcpyDeviceToHost );

    if ( cudaStatus != cudaSuccess ) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, "cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		exit ( 0 );
    }
}

#endif