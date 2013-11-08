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
* <First>       Nov 8, 2013
* <Last>		Nov 8, 2013
* <File>        cudaFunctions.cu
*/

#ifndef __cuda_functions_cpp_
#define __cuda_functions_cpp_


#include "cfdHeaders.h"
#include "macroDef.h"


using namespace std;

static size_t size  = SIM_SIZE;


extern void cudaProject ( float *u, float *v, float *w, float *u0, float *v0, float *w0, dim3 *gridDim, dim3 *blockDim );
extern void cudaAdvect( float *density, float *density0, float *u, float *v, float *w, int boundary, dim3 *gridDim, dim3 *blockDim );
extern void cudaDiffuse ( float *grid, float *grid0, int boundary, float diff, dim3 *gridDim, dim3 *blockDim );
extern void cudaLineSolver (float *grid, float *grid0, int boundary, float a, float c, dim3 *gridDim, dim3 *blockDim);
extern void cudaAddSource ( float *grid, dim3 *gridDim, dim3 *blockDim );
extern void FreeResources ( void );



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