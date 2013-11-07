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
* <Last>		Nov 6, 2013
* <File>        cfdMainKernel.cu
*/

#ifndef __cfd_Main_Kernel_cu_
#define __cfd_Main_Kernel_cu_

#include "macroDef.h"
#include "cudaHelper.h"

using namespace std;

extern void FreeResources(void);

static size_t size  = SIM_SIZE;

#define GetIndex()  \
	int i = blockIdx.x * blockDim.x + threadIdx.x; \
	int j = blockIdx.y * blockDim.y + threadIdx.y; \
	int k = 0; \
	cudaTrans2DTo3D(i, j, k, Grids_X);


__global__ void kernelAddSource ( float *ptr_out )
{
	// Get index of GPU-thread
	GetIndex ( );

	if (i == Grids_X / 2 && j == Grids_X / 2)
	{
		int ind = Index(i, j, k);
		ptr_out[ind] += DELTA_TIME * 1.f;
	}
};

#define is ==
#define gst0   0
#define gstl   Grids_X - 1
#define rsc0   1
#define rscl   SimArea_X

__global__ void kernelSetBoundary ( float *grid_out, int boundary )
{
	// Get index of GPU-thread
	GetIndex ( );

	// Boundary condition
	if ( i >= 1 && i <= SimArea_X && j >= 1 && j <= SimArea_X )
	{
		// Slove line (gst0, y)
		grid_out[Index(gst0, j, k)] = boundary is 1 ? -grid_out[Index(rsc0, j, k)] : grid_out[Index(rsc0, j, k)];
		// Slove line (gstl, y)
		grid_out[Index(gstl, j, k)] = boundary is 1 ? -grid_out[Index(rscl, j, k)] : grid_out[Index(rscl, j, k)];
		// Slove line (x, gst0)
		grid_out[Index(i, gst0, k)] = boundary is 2 ? -grid_out[Index(i, rsc0, k)] : grid_out[Index(i, rsc0, k)];
		// Slove line (x, gstl)
		grid_out[Index(i, gstl, k)] = boundary is 2 ? -grid_out[Index(i, rscl, k)] : grid_out[Index(i, rscl, k)];
	}
	// Slove ghost cell (gst0, gst0)
	grid_out[Index(gst0, gst0, k)] = 0.5f * ( grid_out[Index(rsc0, gst0, k)] + grid_out[Index(gst0, rsc0, k)] );
	// Slove ghost cell (gst0, gstl)
	grid_out[Index(gst0, gstl, k)] = 0.5f * ( grid_out[Index(rsc0, gstl, k)] + grid_out[Index(gst0, rscl, k)] );
	// Slove ghost cell (gstl, gst0)
	grid_out[Index(gstl, gst0, k)] = 0.5f * ( grid_out[Index(rscl, gst0, k)] + grid_out[Index(gstl, rsc0, k)] );
	// Slove ghost cell (gstl, gstl)
	grid_out[Index(gstl, gstl, k)] = 0.5f * ( grid_out[Index(rscl, gstl, k)] + grid_out[Index(gstl, rscl, k)]);

}

#undef gst0
#undef gstl
#undef rsc0
#undef rscl
#undef is


__global__ void kernelLineSolver ( float *grid_inout, float *grid0_in, int boundary, float a, float c )
{
	// Get index of GPU-thread
	GetIndex ( );

	if ( i >= 1 && i <= SimArea_X && j >= 1 && j <= SimArea_X )
	{
		grid_inout[Index(i, j, k)] = (grid0_in[Index(i, j, k)] + a * ( grid_inout[Index(i-1, j, k)] + 
			grid_inout[Index(i+1, j, k)] + grid_inout[Index(i, j-1, k)] + grid_inout[Index(i, j+1,k)] ) ) / c;	
	}
}


__global__ void kernelAdvect ( float *density_out, float *density0_in, float *u_in, float *v_in, float *w_in, float dt0 )
{
	// Get index of GPU-thread
	GetIndex ( );

	int i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1;

	if ( i >= 1 && i <= SimArea_X && j >= 1 && j <= SimArea_X )
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



__global__ void kernelProjectPt1 ( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Get index of GPU-thread
	GetIndex ( );
	
	if ( i >= 1 && i <= SimArea_X && j >= 1 && j <= SimArea_X )
	{
		v0 [ Index (i, j, k) ] = -0.5f * ( u [ Index ( i+1, j, k ) ] - u [ Index ( i-1, j, k ) ] + v [ Index ( i, j+1, k ) ] 
		- v [ Index ( i, j-1, k ) ] ) / SimArea_X;
		u0 [ Index (i, j, k) ] = 0;
	}
}


__global__ void kernelProjectPt2( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Get index of GPU-thread
	GetIndex ( );
	
	if ( i >= 1 && i <= SimArea_X && j >= 1 && j <= SimArea_X )
	{
			u [ Index ( i, j, k ) ] -= 0.5f * SimArea_X * ( u0 [ Index ( i+1, j, k ) ] - u0 [ Index ( i-1, j, k ) ] );
			v [ Index ( i, j, k ) ] -= 0.5f * SimArea_X * ( u0 [ Index ( i, j+1, k ) ] - u0 [ Index ( i, j-1, k ) ] );
	}
}


void cudaAddSource ( float *grid, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	kernelAddSource cudaDevice(*gridDim,  *blockDim) (grid);
};


void cudaLineSolver (float *grid, float *grid0, int boundary, float a, float c, dim3 *gridDim, dim3 *blockDim)
{
    // Launch a kernel on the GPU with one thread for each element.
	for (int i=0; i<20; i++)
	{
		kernelLineSolver cudaDevice(*gridDim,  *blockDim) (grid, grid0, boundary, a, c);
	}
	kernelSetBoundary cudaDevice(*gridDim,  *blockDim)  (grid, boundary);
}


void cudaDiffuse ( float *grid, float *grid0, int boundary, float diff, dim3 *gridDim, dim3 *blockDim )
{
	float a=DELTA_TIME*diff*SimArea_X*SimArea_X;
	cudaLineSolver ( grid, grid0, boundary, a, 1+4*a, gridDim, blockDim );
}


void cudaAdvect( float *density, float *density0, float *u, float *v, float *w, int boundary, dim3 *gridDim, dim3 *blockDim )
{
    // Launch a kernel on the GPU with one thread for each element.
	float dt0 = DELTA_TIME*SimArea_X;
	kernelAdvect      cudaDevice(*gridDim,  *blockDim) ( density, density0, u, v, w, dt0 );
	kernelSetBoundary cudaDevice(*gridDim,  *blockDim) ( density, boundary );
}


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


void DensitySolver ( float *grid, float *grid0, float *u, float *v, float *w )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy ( dev_grid, grid, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_grid0, grid0, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_u, u, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );    
	}

	cudaStatus = cudaMemcpy ( dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_w, w, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }


	cudaAddSource ( dev_grid, &gridDim, &blockDim );
	swap ( dev_grid0, dev_grid ); cudaDiffuse ( dev_grid, dev_grid0, 0, DIFFUSION, &gridDim, &blockDim );
	swap ( dev_grid0, dev_grid ); cudaAdvect  ( dev_grid, dev_grid0, dev_u, dev_v, dev_w, 0, &gridDim, &blockDim );
	
	
	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError ( );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"CUDA encountered an error, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize ( );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaDeviceSynchronize was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy ( grid, dev_grid, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( grid0, dev_grid0, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
	}
	
	cudaStatus = cudaMemcpy ( u, dev_u, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
	}

	cudaStatus = cudaMemcpy ( v, dev_v, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
	}

	cudaStatus = cudaMemcpy ( w, dev_w, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
	}
}


void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );
	
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy ( dev_u0, u0, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_v0, v0, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_w0, w0, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_u, u, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_v, v, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( dev_w, w, size * sizeof(float), cudaMemcpyHostToDevice );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString(cudaStatus));
		FreeResources ( ); exit ( 0 );
    }


	cudaAddSource ( dev_u, &gridDim, &blockDim ); cudaAddSource ( dev_v, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); cudaDiffuse ( dev_u, dev_u0, 1, VISCOSITY, &gridDim, &blockDim );
	swap ( dev_v0, dev_v ); cudaDiffuse ( dev_v, dev_v0, 2, VISCOSITY, &gridDim, &blockDim );
	cudaProject ( dev_u, dev_v, dev_w, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); swap ( dev_v0, dev_v );
	cudaAdvect ( dev_u, dev_u0, dev_u0, dev_v0, dev_w0, 1, &gridDim, &blockDim );
	cudaAdvect ( dev_v, dev_v0, dev_u0, dev_v0, dev_w0, 2, &gridDim, &blockDim );
	cudaProject ( dev_u, dev_v, dev_w, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );


	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError ( );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"CUDA encountered an error, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize ( );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaDeviceSynchronize was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy ( u0, dev_u0, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( v0, dev_v0, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( w0, dev_w0, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }
	
	cudaStatus = cudaMemcpy ( u, dev_u, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND,
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( v, dev_v, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }

	cudaStatus = cudaMemcpy ( w, dev_w, size * sizeof(int), cudaMemcpyDeviceToHost );
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			"cudaMemcpy was failed, at line: %d of file %s", __LINE__, __FILE__ );
		Logfile.SaveStringToFile ( "errormsg.log", sge::SG_FILE_OPEN_APPEND, 
			">>>> Error Message: %s", cudaGetErrorString ( cudaStatus ) );
		FreeResources ( ); exit ( 0 );
    }
}

#endif